#! /usr/bin/env python

"""
Compare two catalogs and report pass/fail for each parameter.
This is designed to make testing new algorithms easier.
"""

import sys
sys.path.insert(0, '.')
from AegeanTools.catalogs import load_table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os


__author__ = 'Paul Hancock'
__date__ = '2015-11-23'


def compare(c1, c2):

    print "# Matching ...",
    s1 = SkyCoord(c1['ra'], c1['dec'], unit=(u.degree, u.degree))
    s2 = SkyCoord(c2['ra'], c2['dec'], unit=(u.degree, u.degree))
    ids2, sep, _ = s1.match_to_catalog_sky(s2)
    mask = np.where(sep < 1.*u.arcmin)
    print " catalog 1 : common : catalog 2 = {0}:{1}:{2}".format(len(c1), len(mask[0]), len(c2))
    c2 = c2[ids2][mask]
    c1 = c1[mask]
    sep = sep[mask]

    print "# Comparing common sources:"
    print "Peak flux ",
    flux_diff = np.abs(c1['peak_flux'] - c2['peak_flux'])
    flux_errs = c1['err_peak_flux']
    result = np.average(flux_diff/flux_errs)
    if result < 1:
        print "({0:5.2e}) PASS ".format(result)
    else:
        print "({0:5.2e}) FAIL".format(result)

    print "Position ",
    pos_errs = np.hypot(c1['err_ra'], c1['err_dec'])
    mask = np.where((c1['err_ra']>0) & (c1['err_dec']>0) & (c2['err_ra']>0) & (c2['err_dec']>0))
    result = np.average(sep[mask]/pos_errs[mask])
    if result < 1:
        print "({0:5.2e}) PASS".format(result)
    else:
        print "({0:5.2e}) FAIL".format(result)

    print "Shape ",
    area_diff = np.abs(c1['a']*c1['b'] - c2['a']*c2['b'])
    area_errs = c1['a']*c1['b']*(c1['err_a'] + c1['err_b'])
    mask = np.where((c1['err_a']>0) & (c1['err_b']>0))
    result = np.average(area_diff[mask]/area_errs[mask])
    if result < 1:
        print "({0:5.2e}) PASS".format(result)
    else:
        print "({0:5.2e}) FAIL".format(result)



if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print "Usage python compare_cat.py catalog1 catalog2"
        sys.exit(1)
    cat1, cat2 = sys.argv[-2:]
    print "Comparing {0} with {1}".format(cat1, cat2)
    c1 = load_table(cat1)
    c2 = load_table(cat2)
    compare(c1, c2)