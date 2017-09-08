#! python
from __future__ import print_function

from AegeanTools import fits_image as fi
from astropy.io import fits
import logging
import numpy as np
from numpy.testing import assert_raises, assert_array_almost_equal

__author__ = 'Paul Hancock'
__date__ = ''

logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)


def test_get_pixinfo():
    header = fits.getheader('tests/test_files/1904-66_SIN.fits')

    area, scale = fi.get_pixinfo(header)
    assert area > 0
    assert len(scale) == 2

    header['CD1_1'] = header['CDELT1']
    del header['CDELT1']
    header['CD2_2'] = header['CDELT2']
    del header['CDELT2']
    area, scale = fi.get_pixinfo(header)
    assert area > 0
    assert len(scale) == 2

    header['CD1_2'] = 0
    header['CD2_1'] = 0
    area, scale = fi.get_pixinfo(header)
    assert area > 0
    assert len(scale) == 2

    header['CD1_2'] = header['CD1_1']
    header['CD2_1'] = header['CD2_2']
    area, scale = fi.get_pixinfo(header)
    assert area == 0
    assert len(scale) == 2

    for f in ['CD1_1', 'CD1_2', 'CD2_2', 'CD2_1']:
        del header[f]
    area, scale = fi.get_pixinfo(header)
    assert area == 0
    assert scale == (0, 0)


def test_get_beam():
    header = fits.getheader('tests/test_files/1904-66_SIN.fits')
    beam = fi.get_beam(header)
    print(beam)
    assert beam is not None
    assert beam.pa == header['BPA']

    del header['BMAJ'], header['BMIN'], header['BPA']
    beam = fi.get_beam(header)
    assert beam is None


def test_fix_aips_header():
    header = fits.getheader('tests/test_files/1904-66_SIN.fits')
    # test when this function is not needed
    newhead = fi.fix_aips_header(header)

    # test when beam params are not present, but there is no aips history
    del header['BMAJ'], header['BMIN'], header['BPA']
    newhead = fi.fix_aips_header(header)

    # test with some aips history
    header['HISTORY'] = 'AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00'
    newhead = fi.fix_aips_header(header)


def test_init():
    filename = 'tests/test_files/1904-66_SIN.fits'
    # normal invocation
    im = fi.FitsImage(filename)

    # call with an already opened hdu instead of a file name
    hdu = fits.open(filename)
    im = fi.FitsImage(hdu)

    # set bzero/bscale
    hdu[0].header['BZERO'] = 1
    hdu[0].header['BSCALE'] = 2
    im = fi.FitsImage(hdu)
    assert im.bscale == 2
    assert im.bzero == 1

    # should be able to supply a beam directly
    beam = fi.Beam(1, 1, 0)
    im = fi.FitsImage(hdu, beam=beam, slice=0)
    assert im.beam is beam

    # raise exception if the beam cannot be determined
    del hdu[0].header['BMAJ']
    assert_raises(Exception, fi.FitsImage, hdu)

    # test with 3 image dimensions
    hdu = fits.open(filename)
    hdu[0].data = np.empty((3, 3, 3))
    # this should fail
    assert_raises(Exception, fi.FitsImage, hdu)
    # this should be fine
    im = fi.FitsImage(hdu, slice=0)
    assert im.x == im.y == 3

    # can't work with 4d data
    hdu[0].data = np.empty((3, 3, 3, 3))
    assert_raises(Exception, fi.FitsImage, hdu)


def test_get_background_rms():
    filename = 'tests/test_files/1904-66_SIN.fits'
    hdu = fits.open(filename)
    hdu[0].data = np.empty((40, 40))
    im = fi.FitsImage(hdu)
    assert im.get_background_rms() > 0


def test_pix2sky_sky2pix():
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
            exec(f+"()")