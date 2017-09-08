#! python
__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import source_finder as sf
from copy import deepcopy
import numpy as np
import logging
import os
import sys

logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)


def test_misc():
    sf.IslandFittingData()
    sf.DummyLM()
    sf.SourceFinder(ignored=None, log=log)


def test_helpers():
    # fix shape
    src = sf.OutputSource()
    src.a = 1
    src.b = 2
    src.pa = 90
    src2 = deepcopy(src)
    sf.fix_shape(src2)
    assert src.a == src2.b
    assert src.b == src2.a
    assert src.pa == src2.pa - 90
    # pa limit
    assert sf.pa_limit(-180.) == 0.
    assert sf.pa_limit(95.) == -85.
    # theta limit
    assert sf.theta_limit(0.) == 0.
    assert sf.theta_limit(np.pi) == 0.
    assert sf.theta_limit(-3*np.pi/2) == np.pi/2
    # scope2lat
    assert sf.scope2lat('MWA') == -26.703319
    assert sf.scope2lat('mwa') == -26.703319
    assert sf.scope2lat('MyFriendsTelescope') is None
    # get_aux
    assert np.all(a is None for a in sf.get_aux_files('_$_fkjfjl'))
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    assert aux_files['rms'] == 'tests/test_files/1904-66_SIN_rms.fits'
    assert aux_files['bkg'] == 'tests/test_files/1904-66_SIN_bkg.fits'
    assert aux_files['mask'] == 'tests/test_files/1904-66_SIN.mim'


def test_load_globals():
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/1904-66_SIN.fits'
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    sfinder.load_globals(filename)
    assert sfinder.global_data.img is not None

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    sfinder.load_globals(filename, bkgin=aux_files['bkg'], rms=1, mask=aux_files['mask'])
    # region isn't available due to healpy not being installed/required
    assert sfinder.global_data.region is not None

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    sfinder.load_globals(filename, rmsin=aux_files['rms'], mask='derp', do_curve=False)
    assert sfinder.global_data.region is None
    img = sfinder._load_aux_image(sfinder.global_data.img, filename)
    assert img is not None

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    from AegeanTools.regions import Region
    sfinder.load_globals(filename, rms=1, mask=Region())
    assert sfinder.global_data.region is not None


def test_find_and_prior_sources():
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/small.fits'
    # vanilla source finding
    found = sfinder.find_sources_in_image(filename, cores=1)
    assert len(found) == 2
    # now with some options
    aux_files = sf.get_aux_files(filename)
    found2 = sfinder.find_sources_in_image(filename, doislandflux=True, outfile=open('dlme', 'w'), nonegative=False,
                                           rmsin=aux_files['rms'], bkgin=aux_files['bkg'],
                                           mask=aux_files['mask'], cores=1, docov=False)
    assert len(found2) == 4
    isle1 = found2[1]
    assert isle1.int_flux > 0
    assert isle1.max_angular_size > 0
    # we should have written some output file
    assert os.path.exists('dlme')
    os.remove('dlme')

    # this should find one less source as one of the source centers is outside the image.
    priorized = sfinder.priorized_fit_islands(filename, catalogue=found, doregroup=False, ratio=1.2, cores=2, docov=False)
    assert len(priorized) == 2
    # this also gives 62 sources even though we turn on regroup
    priorized = sfinder.priorized_fit_islands(filename, catalogue=found, doregroup=True, cores=1, outfile=open('dlme','w'), stage=1)
    assert len(priorized) == 2
    assert len(sfinder.priorized_fit_islands(filename, catalogue=[])) == 0
    # we should have written some output file
    assert os.path.exists('dlme')
    os.remove('dlme')


def test_find_and_prior_parallel():
    log = logging.getLogger("Aegean")
    cores = sf.check_cores(2)
    # don't bother re-running these tests if we have just 1 core
    if cores == 1:
        return
    filename = 'tests/test_files/1904-66_SIN.fits'
    # vanilla source finding
    sfinder = sf.SourceFinder(log=log)
    found = sfinder.find_sources_in_image(filename, cores=cores)
    assert len(found) == 68
    # now with some options
    aux_files = sf.get_aux_files(filename)

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    found2 = sfinder.find_sources_in_image(filename, doislandflux=True, outfile=open('dlme', 'w'), nonegative=False,
                                           rmsin=aux_files['rms'], bkgin=aux_files['bkg'],
                                           mask=aux_files['mask'], cores=cores)
    priorized = sfinder.priorized_fit_islands(filename, catalogue=found, doregroup=True, cores=cores, outfile=open('dlme','w'))
    os.remove('dlme')

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    sfinder.find_sources_in_image('tests/test_files/1904-66_SIN_neg.fits', doislandflux=True, nonegative=False, cores=cores)


def test_save_files():
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/small.fits'
    sfinder.save_background_files(image_filename=filename, outbase='dlme')
    for ext in ['bkg', 'rms', 'snr', 'crv']:
        assert os.path.exists("dlme_{0}.fits".format(ext))
        os.remove("dlme_{0}.fits".format(ext))


def test_save_image():
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/small.fits'
    _ = sfinder.find_sources_in_image(filename, cores=1, max_summits=0, blank=True)
    bfile = 'dlme_blanked.fits'
    sfinder.save_image(bfile)
    assert os.path.exists(bfile)
    os.remove(bfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")