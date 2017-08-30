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


def test_misc():
    a = sf.IslandFittingData()
    b = sf.DummyLM()
    c = sf.SourceFinder(ignored=None, log=log)


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


def test_load_globals():
    logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/1904-66_SIN.fits'
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    sfinder.load_globals(filename)
    assert sfinder.global_data.img is not None
    sfinder.load_globals(filename, bkgin=aux_files['bkg'], rms=1, mask=aux_files['mask'])
    # region isn't available due to healpy not being installed/required
    assert sfinder.global_data.region is None
    sfinder.load_globals(filename, rms=aux_files['rms'], mask='derp', do_curve=False)
    assert sfinder.global_data.region is None
    img = sfinder._load_aux_image(sfinder.global_data.img, filename)
    assert img is not None


def test_find_and_prior_sources():
    logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/1904-66_SIN.fits'
    # vanilla source finding
    found = sfinder.find_sources_in_image(filename, cores=1)
    assert len(found) == 63
    # now with some options
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    found2 = sfinder.find_sources_in_image(filename, doislandflux=True, outfile=open('dlme', 'w'), nonegative=False,
                                           rmsin=aux_files['rms'], bkgin=aux_files['bkg'],
                                           mask=aux_files['mask'], cores=1)
    assert len(found2) == 116
    isle1 = found2[1]
    assert isle1.int_flux > 0
    assert isle1.max_angular_size > 0
    # we should have written some output file
    assert os.path.exists('dlme')
    os.remove('dlme')

    # this should find one less source as one of the source centers is outside the image.
    priorized = sfinder.priorized_fit_islands(filename, catalogue=found, doregroup=False, ratio=1.2, cores=1)
    assert len(priorized) == 62
    # this also gives 62 sources even though we turn on regroup
    priorized = sfinder.priorized_fit_islands(filename, catalogue=found, doregroup=True, cores=1, outfile=open('dlme','w'))
    assert len(priorized) == 62
    assert len(sfinder.priorized_fit_islands(filename, catalogue=[])) == 0
    # we should have written some output file
    assert os.path.exists('dlme')
    os.remove('dlme')


def test_save_files():
    logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/1904-66_SIN.fits'
    sfinder.save_background_files(image_filename=filename, outbase='dlme')
    for ext in ['bkg', 'rms', 'snr', 'crv']:
        assert os.path.exists("dlme_{0}.fits".format(ext))
        os.remove("dlme_{0}.fits".format(ext))


def test_save_image():
    logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/1904-66_SIN.fits'
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