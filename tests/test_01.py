#! python
from __future__ import print_function

from AegeanTools import BANE
from AegeanTools import fits_interp
import numpy as np
import os

__author__ = 'Paul Hancock'
__date__ = '23/08/2017'


def test_sigmaclip():
    # normal usage case
    data = np.random.random(100)
    data[13] = np.nan
    assert len(BANE.sigmaclip(data, 3, 4, reps=4)) > 0

    # test list where all elements get clipped
    assert len(BANE.sigmaclip([-10, 10], 1, 2, reps=2)) == 0

    # test empty list
    assert len(BANE.sigmaclip([], 0, 3)) == 0


def test_fits_interp_compress_then_expand():
    # test compression and expansion
    fits_interp.compress("Test/Images/1904-66_AIT.fits", factor=7, outfile='test.fits')
    fits_interp.expand('test.fits', outfile='test2.fits', method='linear')
    # cleanup
    os.remove('test.fits')
    os.remove('test2.fits')


if __name__ == "__main__":
    test_sigmaclip()