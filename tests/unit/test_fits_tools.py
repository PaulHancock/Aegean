#! /usr/bin/env python
"""
Test fits_interp.py
"""

__author__ = "Paul Hancock"

from AegeanTools import fits_tools
from AegeanTools.wcs_helpers import WCSHelper
from AegeanTools.angle_tools import gcd
from AegeanTools.exceptions import AegeanError
from astropy.io import fits
import numpy as np
import os


def test_load_file_or_hdu():
    """Test that we can 'open' either a file or HDU"""
    fname = "tests/test_files/1904-66_AIT.fits"
    hdulist = fits.open(fname)
    if not (fits_tools.load_file_or_hdu(hdulist) is hdulist):
        raise AssertionError()


def test_compress():
    """Test the compression functionality"""
    # return None when the factor is not a positive integer
    if not (fits_tools.compress(None, factor=-1) is None):
        raise AssertionError()
    if not (fits_tools.compress(None, factor=0.3) is None):
        raise AssertionError()
    if not (fits_tools.compress(None, factor=0) is None):
        raise AssertionError()
    # test with factor = 10 for speed
    fname = "tests/test_files/1904-66_AIT.fits"
    hdulist = fits.open(fname)
    cd1 = hdulist[0].header["CDELT1"]
    cd2 = hdulist[0].header["CDELT2"]
    # compress using CDELT1 and CDELT2
    if not (isinstance(fits_tools.compress(hdulist, factor=10), fits.HDUList)):
        raise AssertionError()
    hdulist[0].header["CD1_1"] = cd1
    del hdulist[0].header["CDELT1"]
    hdulist[0].header["CD2_2"] = cd2
    del hdulist[0].header["CDELT2"]
    # compress using CD1_1 and CDELT2_2 instead
    if not (isinstance(fits_tools.compress(hdulist, factor=10), fits.HDUList)):
        raise AssertionError()
    # now strip CD2_2 and we should get error
    del hdulist[0].header["CD2_2"]
    if not (fits_tools.compress(hdulist, factor=10) is None):
        raise AssertionError()
    # same for CD1_1
    del hdulist[0].header["CD1_1"]
    if not (fits_tools.compress(hdulist, factor=10) is None):
        raise AssertionError()


def test_expand():
    """Test the expand function"""
    fname = "tests/test_files/1904-66_AIT.fits"
    hdulist = fits.open(fname)
    hdulist[0].data[:] = 1.0
    compressed = fits_tools.compress(hdulist, factor=10)
    expanded = fits_tools.expand(compressed)

    # the uncompressed hdu list is missing header keys so test that this gives the expected result
    if not (expanded is hdulist):
        raise AssertionError()

    # ensure that the interpolation isn't completely incorrect
    if not np.all(expanded[0].data == 1.0):
        raise AssertionError("image is not all 1.0")

    # now mix up the CDELT and CD keys
    # reload because we pass references
    compressed = fits_tools.compress(fname, factor=10)
    compressed[0].header["CD1_1"] = compressed[0].header["CDELT1"]
    del compressed[0].header["CDELT1"]
    compressed[0].header["CD2_2"] = compressed[0].header["CDELT2"]
    del compressed[0].header["CDELT2"]
    if not (isinstance(fits_tools.expand(compressed), fits.HDUList)):
        raise AssertionError("CD1_1/CD2_2 doesn't work")

    # now strip CD2_2 and we should return None
    compressed = fits_tools.compress(fname, factor=10)
    del compressed[0].header["CDELT2"]
    if not (fits_tools.expand(compressed) is None):
        raise AssertionError()
    # same for CD1_1
    del compressed[0].header["CDELT1"]
    if not (fits_tools.expand(compressed) is None):
        raise AssertionError()


def test_fits_interp_compress_then_expand():
    """Test that we can interp/compress files"""
    # test compression and expansion
    fits_tools.compress(
        "tests/test_files/1904-66_AIT.fits", factor=7, outfile="tests/temp/test.fits"
    )
    fits_tools.expand("tests/temp/test.fits", outfile="tests/temp/test2.fits")
    # cleanup
    os.remove("tests/temp/test.fits")
    os.remove("tests/temp/test2.fits")


def test_write_fits():
    """Test that we can save a file by providing data and a header"""
    hdu = fits_tools.load_file_or_hdu("tests/test_files/1904-66_SIN.fits")
    header, data = hdu[0].header, hdu[0].data
    outname = "tests/temp/dlme.fits"
    fits_tools.write_fits(data, header, outname)
    if not os.path.exists(outname):
        raise AssertionError("Failed to write data to file {0}".format(outname))
    os.remove(outname)


def test_load_image_band_defaults():
    """Load an image using default values"""
    try:
        data, header = fits_tools.load_image_band("tests/test_files/1904-66_AIT.fits")
    except Exception as e:
        raise e
    if not isinstance(data, np.ndarray):
        raise AssertionError("Loaded data is not an np.ndarray object")
    if not isinstance(header, fits.Header):
        raise AssertionError("header is not a fits.hdu.Header object")
    return


def test_load_image_band_multi_bands():
    """
    Load an image in two bands
    Check that the adjacent bands have adjacent sky coords
    """
    try:
        fits_tools.load_image_band(None, band=(0, 0))
    except AegeanError:
        pass
    else:
        raise AssertionError("Tried to load a total of zero bands")

    try:
        fits_tools.load_image_band(None, band=(1, 1))
    except AegeanError:
        pass
    else:
        raise AssertionError("Tried to load an invalid band combination")

    data0, header0 = fits_tools.load_image_band(
        "tests/test_files/1904-66_SIN.fits", band=(0, 2)
    )
    data1, header1 = fits_tools.load_image_band(
        "tests/test_files/1904-66_SIN.fits", band=(1, 2)
    )
    # The bottom of data0 and top of data1 should be adjacent on the sky
    wcs0 = WCSHelper.from_header(header0)
    pos0 = wcs0.pix2sky((data0.shape[0], data0.shape[1] // 2))
    wcs1 = WCSHelper.from_header(header1)
    pos1 = wcs1.pix2sky((0, data1.shape[1] // 2))

    dist = gcd(*pos0, *pos1)
    if dist > np.hypot(header0["CDELT1"], header0["CDELT2"]) / 2:
        raise AssertionError("adjacent bands don't match up at edges")
    return

def test_load_image_band_cube_as_cube_true():
    """Load an image of a cube with as_cube = True"""
    try:
        data, header = fits_tools.load_image_band("tests/test_files/synthetic_with_alpha.fits", as_cube = True)
    except Exception as e:
        raise e
    if not isinstance(data, np.ndarray):
        raise AssertionError("Loaded data is not an np.ndarray object")
    if not isinstance(header, fits.Header):
        raise AssertionError("header is not a fits.hdu.Header object")
    return

def test_load_image_band_cube_as_cube_false():
    """Load an image of a cube with as_cube = False"""
    try:
        data, header = fits_tools.load_image_band("tests/test_files/synthetic_with_alpha.fits", as_cube = False)
    except Exception as e:
        raise e
    if not isinstance(data, np.ndarray):
        raise AssertionError("Loaded data is not an np.ndarray object")
    if not isinstance(header, fits.Header):
        raise AssertionError("header is not a fits.hdu.Header object")
    return

def test_load_image_band_2d_as_cube_true():
    """Load an image of a band with as_cube = True"""
    try:
        data, header = fits_tools.load_image_band("tests/test_files/1904-66_AIT.fits", as_cube = True)
    except AegeanError as e:
        return e
    else:
        raise AssertionError("Data passed as a cube but only 2 axes were provided")
    

def test_load_image_band_2d_as_cube_false():
    """Load an image using default values"""
    try:
        data, header = fits_tools.load_image_band("tests/test_files/1904-66_AIT.fits", as_cube = False)
    except Exception as e:
        raise e #? Why not use assert?
    if not isinstance(data, np.ndarray):
        raise AssertionError("Loaded data is not an np.ndarray object")
    if not isinstance(header, fits.Header):
        raise AssertionError("header is not a fits.hdu.Header object")
    return

def test_load_image_band_cube_index():
    return


def test_load_image_band_hdu_index():
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
