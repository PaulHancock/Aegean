#! /usr/bin/env python

"""
Compare background and rms images as a test that BANE is working as intended.
"""

__author__ = "Paul Hancock"

from astropy.io import fits
import numpy as np
from scipy.stats import ks_2samp


def compare_pixel_distributions(reference, test):
    """
    Compare the distribution of pixel values in two images.
    Print PASS and return True if the two images have similar enough distributions.
    Print FAIL and return False otherwise.
    If the input images are from BANE, then it is best to use `--compress`-ed images.
    :param reference: filename of first image
    :param test: filename of second image
    :return: True or False
    """
    print "Comparing {0} to {1}".format(reference, test),
    ref = fits.open(reference)[0].data
    ref = ref[np.isfinite(ref)]
    tst = fits.open(test)[0].data
    tst = tst[np.isfinite(tst)]
    stat, p = ks_2samp(ref, tst)
    if p > 0.3 or stat < 0.01:
        print " --> PASS"
        return True
    else:
        print " --> FAIL"
        return False


if __name__ == "__main__":
    # Example usage
    compare_pixel_distributions('1904-66_SIN_bkg.fits', 'Test/Images/1904-66_SIN_bkg_C.fits')
    compare_pixel_distributions('1904-66_SIN_rms.fits', 'Test/Images/1904-66_SIN_rms_C.fits')