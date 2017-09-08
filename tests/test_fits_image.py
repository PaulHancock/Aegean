#! python
from __future__ import print_function

from AegeanTools import fits_image as fi
from astropy.io import fits
import logging

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
    assert beam is not None
    assert beam.pa == header['BPA']

    del header['BMAJ'], header['BMIN'], header['BPA']
    beam = fi.get_beam(header)
    assert beam is None


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")