#! /usr/bin/env python
"""
Test BANE.py
"""
from __future__ import print_function

from AegeanTools import BANE
from astropy.io import fits
import numpy as np
import os

__author__ = 'Paul Hancock'
__date__ = '23/08/2017'

import logging
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")


def test_sigmaclip():
    """Test the sigmaclipping"""
    # normal usage case
    data = np.ones(100)
    if not BANE.sigmaclip(data, 3, 4, reps=4)[0] == 1. :
        raise AssertionError()

    data[13] = np.nan
    if not BANE.sigmaclip(data, 3, 4, reps=4)[0] == 1.:
        raise AssertionError()

    # test empty list
    if not np.isnan(BANE.sigmaclip([], 0, 3)[0]):
        raise AssertionError()


def test_filter_image():
    """Test filter image"""
    # data = np.random.random((30, 30), dtype=np.float32)
    fname = 'tests/test_files/1904-66_SIN.fits'
    outbase = 'dlme'
    rms = outbase + '_rms.fits'
    bkg = outbase + '_bkg.fits'
    # hdu = fits.getheader(fname)
    # shape = hdu[0]['NAXIS1'], hdu[0]['NAXIS2']
    BANE.filter_image(fname, step_size=(10, 10), box_size=(100, 100), cores=2, out_base=outbase)
    if not os.path.exists(rms):
        raise AssertionError()

    os.remove(rms)
    if not os.path.exists(bkg):
        raise AssertionError()

    os.remove(bkg)
    BANE.filter_image(fname, cores=2, out_base=outbase, twopass=True, compressed=True)
    if not os.path.exists(rms):
        raise AssertionError()

    os.remove(rms)
    if not os.path.exists(bkg):
        raise AssertionError()

    os.remove(bkg)


def test_ND_images():
    """Test that ND images are treated correctly"""
    fbase = 'tests/test_files/small_{0}D.fits'
    outbase = 'dlme'
    rms = outbase + '_rms.fits'
    bkg = outbase + '_bkg.fits'
    # this should work just fine, but trigger different NAXIS checks
    for fname in [fbase.format(n) for n in [3,4]]:
        BANE.filter_image(fname, out_base=outbase)
        os.remove(rms)
        os.remove(bkg)

    fname = fbase.format(5)
    try:
        BANE.filter_image(fname,out_base=outbase)
    except Exception as e:
        pass
    else:
        raise AssertionError()
    

def test_quantitative():
    """Test that the images are equal to a pre-calculated version"""
    fbase = 'tests/test_files/1904-66_SIN'
    outbase = 'dlme'
    BANE.filter_image(fbase+'.fits', out_base=outbase, cores=2, nslice=2)

    rms = outbase + '_rms.fits'
    bkg = outbase + '_bkg.fits'
    ref_rms = fbase + '_rms.fits'
    ref_bkg = fbase + '_bkg.fits'

    r1 = fits.getdata(rms)
    r2 = fits.getdata(ref_rms)
    b1 = fits.getdata(bkg)
    b2 = fits.getdata(ref_bkg)
    os.remove(rms)
    os.remove(bkg)

    if not np.allclose(r1, r2, atol=0.01, equal_nan=True):
        raise AssertionError("rms is wrong")

    if not np.allclose(b1, b2, atol=0.003, equal_nan=True):
        raise AssertionError("bkg is wrong")

    return


def test_BSCALE():
    """Test that BSCALE present and not 1.0 is handled properly"""
    fbase = 'tests/test_files/1904-66_SIN'
    outbase = 'dlme'
    hdu = fits.open(fbase+'.fits')
    hdu[0].header['BSCALE'] = 1.0
    hdu.writeto('dlme.fits')
    try:
        BANE.filter_image(outbase+'.fits', out_base=outbase, cores=1, nslice=1)
    except ValueError as e:
        raise AssertionError("BSCALE=1.0 causes crash")
    finally:
        os.remove('dlme.fits')

    hdu[0].header['BSCALE'] = 2.0
    hdu.writeto('dlme.fits')
    try:
        BANE.filter_image(outbase+'.fits', out_base=outbase, cores=1, nslice=1)
    except ValueError as e:
        raise AssertionError("BSCALE=2.0 causes crash")
    finally:
        os.remove('dlme.fits')
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
