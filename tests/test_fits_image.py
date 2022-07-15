#! /usr/bin/env python
"""
Test fits_image.py
"""

import logging

import AegeanTools.wcs_helpers
import numpy as np
from AegeanTools import fits_image as fi
from astropy.io import fits
from numpy.testing import assert_array_almost_equal, assert_raises

__author__ = 'Paul Hancock'

logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)


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
    if not im.bscale == 2:
        raise AssertionError()
    if not im.bzero == 1:
        raise AssertionError()

    # should be able to supply a beam directly
    beam = AegeanTools.wcs_helpers.Beam(1, 1, 0)
    im = fi.FitsImage(hdu, beam=beam, cube_index=0)
    if not (im.beam is beam):
        raise AssertionError()

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
    if not (im.x == im.y == 3):
        raise AssertionError()

    # can't work with 4d data
    hdu[0].data = np.empty((3, 3, 3, 3))
    assert_raises(Exception, fi.FitsImage, hdu)


def test_get_background_rms():
    """Test get_background_rms"""
    filename = 'tests/test_files/1904-66_SIN.fits'
    hdu = fits.open(filename)
    hdu[0].data = np.empty((40, 40))
    im = fi.FitsImage(hdu)
    if not (im.get_background_rms() > 0):
        raise AssertionError()


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
