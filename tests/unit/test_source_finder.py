#! /usr/bin/env python
"""
Test source_finder.py
"""

__author__ = "Paul Hancock"

import os
from copy import deepcopy

import numpy as np
from astropy.io import fits

from AegeanTools import models
from AegeanTools import source_finder as sf
from AegeanTools.exceptions import AegeanError
from AegeanTools.logging import logger
from AegeanTools.models import classify_catalog
from AegeanTools.regions import Region


def test_psf_with_nans():
    """Test that a psf map with nans doesn't create a crash"""
    sfinder = sf.SourceFinder()
    filename = "tests/test_files/synthetic_test.fits"
    psf = "tests/test_files/synthetic_test_psf.fits"
    # create a test psf map with all major axis being nans
    hdu = fits.open(psf)
    print(hdu[0].data.shape)
    hdu[0].data[0, :, :] = np.nan
    hdu.writeto("dlme_psf.fits", overwrite=True)

    try:
        _ = sfinder.find_sources_in_image(
            filename, cores=1, rms=0.5, bkg=0, imgpsf="dlme_psf.fits"
        )
    except AssertionError as e:
        os.remove("dlme_psf.fits")
        if "major" in e.args[0]:
            raise AssertionError("Broken on psf maps with nans")
        else:
            raise
    else:
        os.remove("dlme_psf.fits")
    return


def test_misc():
    """Test some random things"""
    sf.IslandFittingData()
    sf.DummyLM()
    sf.SourceFinder(ignored=None)


def test_helpers():
    """Test the helper functions"""
    # fix shape
    src = sf.ComponentSource()
    src.a = 1
    src.b = 2
    src.pa = 90
    src2 = deepcopy(src)
    sf.fix_shape(src2)
    if not (src.a == src2.b):
        raise AssertionError()
    if not (src.b == src2.a):
        raise AssertionError()
    if not (src.pa == src2.pa - 90):
        raise AssertionError()
    # pa limit
    if not (sf.pa_limit(-180.0) == 0.0):
        raise AssertionError()
    if not (sf.pa_limit(95.0) == -85.0):
        raise AssertionError()
    # theta limit
    if not (sf.theta_limit(0.0) == 0.0):
        raise AssertionError()
    if not (sf.theta_limit(np.pi) == 0.0):
        raise AssertionError()
    if not (sf.theta_limit(-3 * np.pi / 2) == np.pi / 2):
        raise AssertionError()
    # get_aux
    if not (np.all(a is None for a in sf.get_aux_files("_$_fkjfjl"))):
        raise AssertionError()
    aux_files = sf.get_aux_files("tests/test_files/1904-66_SIN.fits")
    if not (aux_files["rms"] == "tests/test_files/1904-66_SIN_rms.fits"):
        raise AssertionError()
    if not (aux_files["bkg"] == "tests/test_files/1904-66_SIN_bkg.fits"):
        raise AssertionError()
    if not (aux_files["mask"] == "tests/test_files/1904-66_SIN.mim"):
        raise AssertionError()


def test__make_bkg_rms():
    """Ensure that SourceFinder._make_bkg_rms works properly"""
    sfinder = sf.SourceFinder()
    filename = "tests/test_files/1904-66_SIN.fits"
    sfinder.load_globals(filename)

    # check that we don't make mistake #163 again.
    if not np.all(sfinder.rmsimg[50:55, 50:55] > 0):
        raise AssertionError("RMS map is not positive in the middle")
    if not np.any(sfinder.bkgimg[50:55, 50:55] != 0):
        raise AssertionError("BKG map is all zero in the middle")


def test_load_globals():
    """Test load_globals"""
    sfinder = sf.SourceFinder()
    filename = "tests/test_files/1904-66_SIN.fits"
    aux_files = sf.get_aux_files("tests/test_files/1904-66_SIN.fits")
    sfinder.load_globals(filename)
    if sfinder.img is None:
        raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder()
    sfinder.load_globals(
        filename, bkgin=aux_files["bkg"], rms=1, mask=aux_files["mask"]
    )
    # region isn't available due to healpy not being installed/required
    if sfinder.region is None:
        raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder()
    sfinder.load_globals(
        filename, bkgin=aux_files["bkg"], bkg=0, mask=aux_files["mask"]
    )
    # region isn't available due to healpy not being installed/required
    if sfinder.region is None:
        raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder()
    sfinder.load_globals(
        filename, bkgin=aux_files["bkg"], rms=1, bkg=0, mask=aux_files["mask"]
    )
    # region isn't available due to healpy not being installed/required
    if sfinder.region is None:
        raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder()
    sfinder.load_globals(filename, rmsin=aux_files["rms"], do_curve=False, mask="derp")
    if sfinder.region is not None:
        raise AssertionError()
    img = sfinder._load_aux_image(sfinder.img, filename)
    if img is None:
        raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder()
    aux_files = sf.get_aux_files("tests/test_files/1904-66_SIN.fits")
    from AegeanTools.regions import Region

    sfinder.load_globals(filename, rms=1, mask=Region())
    if sfinder.region is None:
        raise AssertionError()
    
def test_load_globals_cube():
    """Test load_globals"""
    sfinder = sf.SourceFinder()
    filename = "tests/test_files/synthetic_with_alpha.fits"
    # aux_files = sf.get_aux_files("tests/test_files/synthetic_with_alpha.fits")
    bkg = filename
    rms = filename
    sfinder.load_globals(filename, bkgin=bkg, rmsin= rms, as_cube=True)
    if sfinder.img is None:
        raise AssertionError()
    if np.allclose(sfinder.img, 0):
        print("True")
    else:
        raise AssertionError()

def test_find_and_prior_sources():
    """Test find sources and prior sources"""
    try:
        sfinder = sf.SourceFinder()
        filename = "tests/test_files/synthetic_test.fits"
        nsrc = 98
        nisl = 97
        ntot = nsrc + nisl

        # vanilla source finding
        found = sfinder.find_sources_in_image(filename, cores=1, rms=0.5, bkg=0)
        if not (len(found) == nsrc):
            raise AssertionError(
                f"Vanilla source finding: wrong number of sources {len(found)}, expecting {nsrc}"
            )

        # source finding but not fitting
        found = sfinder.find_sources_in_image(
            filename, cores=1, max_summits=0, rms=0.5, bkg=0
        )
        if not (len(found) == (nsrc)):
            raise AssertionError(
                f"Finding not fitting: wrong number of sources {len(found)}, expecting {nsrc}"
            )

        # now with some options
        aux_files = sf.get_aux_files(filename)
        found2 = sfinder.find_sources_in_image(
            filename,
            doislandflux=True,
            outfile=open("dlme", "w"),
            nonegative=False,
            rmsin=aux_files["rms"],
            bkgin=aux_files["bkg"],
            mask=aux_files["mask"],
            cores=1,
            docov=False,
        )
        if not (len(found2) == ntot):
            raise AssertionError(
                f"Fitting w aux files: wrong number of sources {len(found2)}, expecting {ntot}"
            )
        isle1 = found2[1]
        if not (isle1.int_flux > 0):
            raise AssertionError()
        if not (isle1.max_angular_size > 0):
            raise AssertionError()
        # we should have written some output file
        if not (os.path.exists("dlme")):
            raise AssertionError()
        os.remove("dlme")

        # some more tests, now using multiple cores
        cores = 2

        priorized = sfinder.priorized_fit_islands(
            filename,
            catalogue=found,
            doregroup=False,
            ratio=1.2,
            cores=cores,
            rmsin=aux_files["rms"],
            bkgin=aux_files["bkg"],
            docov=False,
        )
        if not (len(priorized) == (nsrc)):
            raise AssertionError(
                f"Multi cores: wrong number of sources {len(priorized)}, expecting {nsrc}"
            )

        priorized = sfinder.priorized_fit_islands(
            filename,
            catalogue=found,
            doregroup=True,
            cores=1,
            rmsin=aux_files["rms"],
            bkgin=aux_files["bkg"],
            outfile=open("dlme", "w"),
            stage=1,
        )
        if not (len(priorized) == (nsrc)):
            raise AssertionError(
                f"Found the wrong number of sources {len(priorized)}, expecting {nsrc}"
            )
        if not (len(sfinder.priorized_fit_islands(filename, catalogue=[])) == 0):
            raise AssertionError()
        # we should have written some output file
        if not (os.path.exists("dlme")):
            raise AssertionError("Failed to create output file")
    finally:
        os.remove("dlme")


def dont_test_find_and_prior_parallel():
    """Test find/piroirze with parallel operation"""
    cores = 1

    filename = "tests/test_files/synthetic_test.fits"
    # vanilla source finding
    logger.info("basic fitting (no bkg/rms")
    sfinder = sf.SourceFinder()
    found = sfinder.find_sources_in_image(filename, cores=cores, bkg=0, rms=0.5)
    if not (len(found) == 98):
        raise AssertionError("found {0} sources".format(len(found)))
    # now with some options
    aux_files = sf.get_aux_files(filename)

    del sfinder
    logger.info("fitting with supplied bkg/rms and 2 cores")
    cores = 2
    sfinder = sf.SourceFinder()
    _ = sfinder.find_sources_in_image(
        filename,
        doislandflux=True,
        outfile=open("dlme", "w"),
        nonegative=False,
        rmsin=aux_files["rms"],
        bkgin=aux_files["bkg"],
        mask=aux_files["mask"],
        cores=cores,
    )

    logger.info("now priorised fitting")
    _ = sfinder.priorized_fit_islands(
        filename,
        catalogue=found,
        doregroup=True,
        cores=cores,
        outfile=open("dlme", "w"),
    )
    os.remove("dlme")

    del sfinder
    logger.info("fitting negative sources")
    sfinder = sf.SourceFinder()
    sfinder.find_sources_in_image(
        "tests/test_files/1904-66_SIN_neg.fits",
        doislandflux=True,
        nonegative=False,
        cores=cores,
    )


def test_save_files():
    """Test that we can save files"""
    sfinder = sf.SourceFinder()
    filename = "tests/test_files/small.fits"
    sfinder.save_background_files(image_filename=filename, outbase="dlme")
    for ext in ["bkg", "rms", "snr", "crv"]:
        if not (os.path.exists("dlme_{0}.fits".format(ext))):
            raise AssertionError()
        os.remove("dlme_{0}.fits".format(ext))


def test_save_image():
    """Test save_image"""
    sfinder = sf.SourceFinder()
    filename = "tests/test_files/small.fits"
    _ = sfinder.find_sources_in_image(filename, cores=1, max_summits=0, blank=True)
    bfile = "dlme_blanked.fits"
    sfinder.save_image(bfile)
    if not (os.path.exists(bfile)):
        raise AssertionError()
    os.remove(bfile)


def test_esimate_lmfit_parinfo():
    """Test estimate_lmfit_parinfo"""
    sfinder = sf.SourceFinder()

    data = np.zeros(shape=(3, 3))
    rmsimg = np.ones(shape=(3, 3))

    # should hit isnegative
    data[1, 1] = -6
    # should hit outerclip is None
    outerclip = None
    # should run error because curve is the wrong shape
    curve = np.zeros((3, 4))
    try:
        sfinder.estimate_lmfit_parinfo(
            data=data, rmsimg=rmsimg, curve=curve, innerclip=5, outerclip=outerclip
        )
    except AssertionError as e:
        e.message = "Passed"
    else:
        raise AssertionError(
            "estimate_lmfit_parinfo should err when curve.shape != data.shape"
        )

    return


def test_island_contours():
    """Test that island contours are correct"""
    sfinder = sf.SourceFinder()
    filename = "tests/test_files/synthetic_test.fits"

    # vanilla source finding
    found = sfinder.find_sources_in_image(
        filename, cores=1, rms=0.5, bkg=0, doislandflux=True
    )

    components, islands, simples = classify_catalog(found)
    isle_0_contour = np.array(
        [
            (41, 405),
            (41, 406),
            (41, 407),
            (42, 407),
            (42, 408),
            (42, 409),
            (43, 409),
            (43, 410),
            (44, 410),
            (45, 410),
            (46, 410),
            (47, 410),
            (47, 409),
            (48, 409),
            (48, 408),
            (49, 408),
            (49, 407),
            (49, 406),
            (49, 405),
            (48, 405),
            (48, 404),
            (48, 403),
            (47, 403),
            (46, 403),
            (45, 403),
            (44, 403),
            (43, 403),
            (43, 404),
            (42, 404),
            (42, 405),
        ]
    )
    if not np.all(np.array(islands[0].contour) == isle_0_contour):
        raise AssertionError("Island contour for island 0 is incorrect")
    return


# for 3.0 functionality


def test_find_islands():
    im = np.ones((10, 12), dtype=np.float32)
    bkg = np.zeros_like(im)
    rms = np.ones_like(im)

    # test with no islands and no logger
    islands = sf.find_islands(im, bkg, rms)
    if len(islands) != 0:
        return AssertionError("Found islands where none existed")

    # now set just one island
    im[3:6, 4:7] *= 10
    # and have some pixels masked or below the clipping threshold
    im[6, 5] = np.nan
    im[4, 4] = 0
    # make the border nans
    im[0:3, :] = im[-1:, :] = np.nan
    im[:, 0] = im[:, -1] = np.nan

    islands = sf.find_islands(im, bkg, rms)

    if len(islands) != 1:
        raise AssertionError(
            "Incorrect number of islands found {0}, expecting 1".format(len(islands))
        )
    if not isinstance(islands[0], models.PixelIsland):
        raise AssertionError(
            "Islands[0] is not a PixelIsland but instead a {0}".format(type(islands[0]))
        )

    correct_box = [[3, 6], [4, 7]]
    if not np.all(islands[0].bounding_box == correct_box):
        raise AssertionError(
            "Bounding box incorrect, should be {0}, but is {1}".format(
                correct_box, islands[0].bounding_box
            )
        )

    # add another island that is between the seed/flood thresholds
    im[7:9, 2:5] = 4.5
    islands = sf.find_islands(im, bkg, rms)
    if len(islands) != 1:
        raise AssertionError(
            "Incorrect number of islands found {0}, expecting 1".format(len(islands))
        )

    return


def test_regions_used_in_finding():
    """Ensure that a region is use appropriately in source finding"""
    imfile = "tests/test_files/1904-66_SIN.fits"
    aux_files = sf.get_aux_files(imfile)
    # region outside of the test image
    reg1 = Region()
    reg1.add_circles(np.radians(10), np.radians(20), np.radians(1))
    # region that contains the test image
    reg2 = Region()
    reg2.add_circles(np.radians(286), np.radians(-66), np.radians(10))

    sfinder = sf.SourceFinder()
    sources = sfinder.find_sources_in_image(
        imfile, rmsin=aux_files["rms"], bkgin=aux_files["bkg"], cores=1, mask=reg1
    )
    if len(sources) > 0:
        raise AssertionError("Found sources outside of region specified.")

    del sfinder, sources
    sfinder = sf.SourceFinder()
    sources = sfinder.find_sources_in_image(
        imfile, rmsin=aux_files["rms"], bkgin=aux_files["bkg"], cores=1, mask=reg2
    )
    if len(sources) == 0:
        print(len(sources))
        raise AssertionError("Failed to find any islands within region specified.")

    return


def test_load_compressed_aux_files():
    """
    Test against issue #193: aegean failing to load bkg/rms files that were
    created with BANE --compress
    """
    background = "tests/test_files/1904-66_bkg_compressed.fits"
    noise = "tests/test_files/1904-66_rms_compressed.fits"
    image = "tests/test_files/1904-66_SIN.fits"

    sfinder = sf.SourceFinder()
    try:
        sfinder.load_globals(image, bkgin=background, rmsin=noise)
    except AegeanError as ae:
        raise AssertionError(ae)
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
            print("... PASS")
