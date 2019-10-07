#! /usr/bin/env python
"""
Test fits_image.py
"""

from __future__ import print_function

import AegeanTools.wcs_helpers

__author__ = 'Paul Hancock'

from AegeanTools import fits_image as fi
from astropy.io import fits
import logging
import numpy as np
from numpy.testing import assert_raises, assert_array_almost_equal

logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)


def test_get_pixinfo():
    """Test that we can get info from various header styles"""
    header = fits.getheader('tests/test_files/1904-66_SIN.fits')

    area, scale = AegeanTools.wcs_helpers.get_pixinfo(header)
    if not area > 0: raise AssertionError()
    if not len(scale) == 2: raise AssertionError()

    header['CD1_1'] = header['CDELT1']
    del header['CDELT1']
    header['CD2_2'] = header['CDELT2']
    del header['CDELT2']
    area, scale = AegeanTools.wcs_helpers.get_pixinfo(header)
    if not area > 0: raise AssertionError()
    if not len(scale) == 2: raise AssertionError()

    header['CD1_2'] = 0
    header['CD2_1'] = 0
    area, scale = AegeanTools.wcs_helpers.get_pixinfo(header)
    if not area > 0: raise AssertionError()
    if not len(scale) == 2: raise AssertionError()

    header['CD1_2'] = header['CD1_1']
    header['CD2_1'] = header['CD2_2']
    area, scale = AegeanTools.wcs_helpers.get_pixinfo(header)
    if not area == 0: raise AssertionError()
    if not len(scale) == 2: raise AssertionError()

    for f in ['CD1_1', 'CD1_2', 'CD2_2', 'CD2_1']:
        del header[f]
    area, scale = AegeanTools.wcs_helpers.get_pixinfo(header)
    if not area == 0: raise AssertionError()
    if not scale == (0, 0): raise AssertionError()


def test_get_beam():
    """Test that we can recover the beam from the fits header"""
    header = fits.getheader('tests/test_files/1904-66_SIN.fits')
    beam = AegeanTools.wcs_helpers.get_beam(header)
    print(beam)
    if beam is None : raise AssertionError()
    if beam.pa != header['BPA']: raise AssertionError()

    del header['BMAJ'], header['BMIN'], header['BPA']
    beam = AegeanTools.wcs_helpers.get_beam(header)
    if beam is not None : raise AssertionError()


def test_fix_aips_header():
    """TEst that we can fix an aips generated fits header"""
    header = fits.getheader('tests/test_files/1904-66_SIN.fits')
    # test when this function is not needed
    _ = AegeanTools.wcs_helpers.fix_aips_header(header)

    # test when beam params are not present, but there is no aips history
    del header['BMAJ'], header['BMIN'], header['BPA']
    _ = AegeanTools.wcs_helpers.fix_aips_header(header)

    # test with some aips history
    header['HISTORY'] = 'AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00'
    _ = AegeanTools.wcs_helpers.fix_aips_header(header)


def test_init():
    """Test that FitsImage __init__ works """
    filename = 'tests/test_files/1904-66_SIN.fits'
    # normal invocation
    _ = fi.FitsImage(filename)

    # call with an already opened hdu instead of a file name
    hdu = fits.open(filename)
    _ = fi.FitsImage(hdu)

    # set bzero/bscale
    hdu[0].header['BZERO'] = 1
    hdu[0].header['BSCALE'] = 2
    im = fi.FitsImage(hdu)
    if not im.bscale == 2: raise AssertionError()
    if not im.bzero == 1: raise AssertionError()

    # should be able to supply a beam directly
    beam = AegeanTools.wcs_helpers.Beam(1, 1, 0)
    im = fi.FitsImage(hdu, beam=beam, cube_index=0)
    if not (im.beam is beam): raise AssertionError()

    # raise exception if the beam cannot be determined
    del hdu[0].header['BMAJ']
    assert_raises(Exception, fi.FitsImage, hdu)

    # test with 3 image dimensions
    hdu = fits.open(filename)
    hdu[0].data = np.empty((3, 3, 3))
    # this should fail
    assert_raises(Exception, fi.FitsImage, hdu)
    # this should be fine
    im = fi.FitsImage(hdu, cube_index=0)
    if not (im.x == im.y == 3): raise AssertionError()

    # can't work with 4d data
    hdu[0].data = np.empty((3, 3, 3, 3))
    assert_raises(Exception, fi.FitsImage, hdu)


def test_get_background_rms():
    """Test get_background_rms"""
    filename = 'tests/test_files/1904-66_SIN.fits'
    hdu = fits.open(filename)
    hdu[0].data = np.empty((40, 40))
    im = fi.FitsImage(hdu)
    if not (im.get_background_rms() > 0): raise AssertionError()


def test_pix2sky_sky2pix():
    """Test pix2sky and sky2pix are conjugate"""
    filename = 'tests/test_files/1904-66_SIN.fits'
    hdu = fits.open(filename)
    im = fi.FitsImage(hdu)
    ra, dec = im.pix2sky([0, 0])
    x, y = im.sky2pix([ra, dec])
    assert_array_almost_equal([0, 0], [x, y])


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()