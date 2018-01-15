#! python
__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import source_finder as sf
from copy import deepcopy
import numpy as np
import logging
import os
import six

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
    if not (src.a == src2.b): raise AssertionError()
    if not (src.b == src2.a): raise AssertionError()
    if not (src.pa == src2.pa - 90): raise AssertionError()
    # pa limit
    if not (sf.pa_limit(-180.) == 0.): raise AssertionError()
    if not (sf.pa_limit(95.) == -85.): raise AssertionError()
    # theta limit
    if not (sf.theta_limit(0.) == 0.): raise AssertionError()
    if not (sf.theta_limit(np.pi) == 0.): raise AssertionError()
    if not (sf.theta_limit(-3*np.pi/2) == np.pi/2): raise AssertionError()
    # scope2lat
    if not (sf.scope2lat('MWA') == -26.703319): raise AssertionError()
    if not (sf.scope2lat('mwa') == -26.703319): raise AssertionError()
    if sf.scope2lat('MyFriendsTelescope') is not None: raise AssertionError()
    # get_aux
    if not (np.all(a is None for a in sf.get_aux_files('_$_fkjfjl'))): raise AssertionError()
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    if not (aux_files['rms'] == 'tests/test_files/1904-66_SIN_rms.fits'): raise AssertionError()
    if not (aux_files['bkg'] == 'tests/test_files/1904-66_SIN_bkg.fits'): raise AssertionError()
    if not (aux_files['mask'] == 'tests/test_files/1904-66_SIN.mim'): raise AssertionError()


def test_load_globals():
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/1904-66_SIN.fits'
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    sfinder.load_globals(filename)
    if sfinder.global_data.img is None: raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    sfinder.load_globals(filename, bkgin=aux_files['bkg'], rms=1, mask=aux_files['mask'])
    # region isn't available due to healpy not being installed/required
    if sfinder.global_data.region is None: raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    sfinder.load_globals(filename, rmsin=aux_files['rms'], mask='derp', do_curve=False)
    if sfinder.global_data.region is not None: raise AssertionError()
    img = sfinder._load_aux_image(sfinder.global_data.img, filename)
    if img is None: raise AssertionError()

    del sfinder
    sfinder = sf.SourceFinder(log=log)
    aux_files = sf.get_aux_files('tests/test_files/1904-66_SIN.fits')
    from AegeanTools.regions import Region
    sfinder.load_globals(filename, rms=1, mask=Region())
    if sfinder.global_data.region is None: raise AssertionError()


def test_find_and_prior_sources():
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/small.fits'
    # vanilla source finding
    found = sfinder.find_sources_in_image(filename, cores=1)
    if not (len(found) == 2): raise AssertionError()
    # now with some options
    aux_files = sf.get_aux_files(filename)
    found2 = sfinder.find_sources_in_image(filename, doislandflux=True, outfile=open('dlme', 'w'), nonegative=False,
                                           rmsin=aux_files['rms'], bkgin=aux_files['bkg'],
                                           mask=aux_files['mask'], cores=1, docov=False)
    if not (len(found2) == 4): raise AssertionError()
    isle1 = found2[1]
    if not (isle1.int_flux > 0): raise AssertionError()
    if not (isle1.max_angular_size > 0): raise AssertionError()
    # we should have written some output file
    if not (os.path.exists('dlme')): raise AssertionError()
    os.remove('dlme')

    # pprocess is broken in python3 at the moment so just use 1 core.
    if six.PY3:
        cores = 1
    else:
        cores = 2
    # this should find one less source as one of the source centers is outside the image.
    priorized = sfinder.priorized_fit_islands(filename, catalogue=found, doregroup=False, ratio=1.2, cores=cores, docov=False)
    if not (len(priorized) == 2): raise AssertionError()
    # this also gives 62 sources even though we turn on regroup
    priorized = sfinder.priorized_fit_islands(filename, catalogue=found, doregroup=True, cores=1, outfile=open('dlme','w'), stage=1)
    if not (len(priorized) == 2): raise AssertionError()
    if not (len(sfinder.priorized_fit_islands(filename, catalogue=[])) == 0): raise AssertionError()
    # we should have written some output file
    if not (os.path.exists('dlme')): raise AssertionError()
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
    if not (len(found) == 68): raise AssertionError()
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
        if not (os.path.exists("dlme_{0}.fits".format(ext))): raise AssertionError()
        os.remove("dlme_{0}.fits".format(ext))


def test_save_image():
    log = logging.getLogger("Aegean")
    sfinder = sf.SourceFinder(log=log)
    filename = 'tests/test_files/small.fits'
    _ = sfinder.find_sources_in_image(filename, cores=1, max_summits=0, blank=True)
    bfile = 'dlme_blanked.fits'
    sfinder.save_image(bfile)
    if not (os.path.exists(bfile)): raise AssertionError()
    os.remove(bfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()