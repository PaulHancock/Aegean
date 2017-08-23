#! python
from __future__ import print_function

from AegeanTools import BANE
from AegeanTools import fits_interp
import numpy as np
import os

__author__ = 'Paul Hancock'
__date__ = '23/08/2017'


def test_sigmaclip():
    data = np.random.random(100)
    print("TESTING BANE.sigmaclip")
    BANE.sigmaclip(data, 3, 4, reps=4)
    print("Pass")


def test_fits_interp_compress_then_expand():
    # test compression and expansion
    fits_interp.compress("Test/Images/1904-66_AIT.fits", factor=7, outfile='test.fits')
    fits_interp.expand('test.fits', outfile='test2.fits', method='linear')
    # cleanup
    os.remove('test.fits')
    os.remove('test2.fits')


if __name__ == "__main__":
    test_sigmaclip()