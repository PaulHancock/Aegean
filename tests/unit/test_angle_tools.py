#! /usr/bin/env python
"""
Test the angle_tools module
"""

import astropy.units as u
import numpy as np
from treasure_island import angle_tools as at
from astropy.coordinates import Angle
from numpy.testing import assert_almost_equal, assert_approx_equal

__author__ = 'Paul Hancock'


def test_ra2dec():
    """Test ra2dec against astropy conversion"""
    # Test against the astropy calculations
    for ra in ['14:21:45.003', '-12 04 22', '-00 01 12.003']:
        ans = at.ra2dec(ra)
        desired = Angle(ra, unit=u.hourangle).hour * 15
        assert_approx_equal(ans, desired, "{0} != {1}".format(ans, desired))


def test_dec2dec():
    """Test dec2dec against astropy conversion"""
    # Test against the astropy calculations
    for dec in ['+14:21:45.003', '-99 04 22', '-00 01 23.456', '00 01']:
        ans = at.dec2dec(dec)
        desired = Angle(dec, unit=u.degree).degree
        assert_approx_equal(
            ans, desired, err_msg="{0} != {1}".format(ans, desired))


def test_dec2dms():
    """Test conversion of dec to DMS strings"""
    for dec, dstr in [(-0.12345, "-00:07:24.42"),
                      (80.0, "+80:00:00.00"),
                      (np.nan, "XX:XX:XX.XX"),
                      (np.inf, "XX:XX:XX.XX")]:
        ans = at.dec2dms(dec)
        if not ans == dstr:
            raise AssertionError("{0} != {1}".format(ans, dstr))


def test_dec2hms():
    """Test conversion of RA to HMS strings"""
    for dec, dstr in [(-15, "23:00:00.00"),
                      (15, "01:00:00.00"),
                      (23.5678, "01:34:16.27"),
                      (np.nan, "XX:XX:XX.XX"),
                      (np.inf, "XX:XX:XX.XX")]:
        ans = at.dec2hms(dec)
        if not ans == dstr:
            raise AssertionError("{0} != {1}".format(ans, dstr))


def test_gcd():
    """Test the calculation of great circle distance"""
    for ra1, dec1, ra2, dec2, dist in [(0, 0, 0, 1, 1),  # simple 1 deg offset
                                       (0, -90, 180, 90, 180),  # pole to pole
                                       (120, 89, 300, 89, 2.),  # over the pole
                                       # distances very close to 180deg
                                       (0, 0, 179.99999, 0, 179.99999),
                                       # at the south pole
                                       (12.0, -90, 45, -90, 0)
                                       ]:
        ans = at.gcd(ra1, dec1, ra2, dec2)
        assert_almost_equal(
            ans, dist, err_msg="{0:5.2f},{1:5.2f} <-> {2:5.2f},{3:5.2f} == {4:g} != {5:g}".format(ra1, dec1, ra2, dec2, dist, ans))


def test_bear():
    """Test bearing calculation"""
    for ra1, dec1, ra2, dec2, bear in [(0, 0, 0, 1, 0),
                                       (0, 0, 180, 90, 0),
                                       (0, 0, 179.99999, 0, 90),
                                       (0, 0, 180.00001, 0, -90)
                                       ]:
        ans = at.bear(ra1, dec1, ra2, dec2)
        assert_almost_equal(
            ans, bear, err_msg="{0:5.2f},{1:5.2f} <-> {2:5.2f},{3:5.2f} == {4:g} != {5:g}".format(ra1, dec1, ra2, dec2, bear, ans))


def test_translate():
    """Test the translate function"""
    for (ra1, dec1), (r, theta), (ra2, dec2) in [((0, 0), (1, 0), (0, 1)),
                                                 # over the pole
                                                 ((45, 89.75), (0.5, 0),
                                                  (225, 89.75)),
                                                 # negative r
                                                 ((12, -45), (-1, 180), (12, -44))
                                                 ]:
        ans = at.translate(ra1, dec1, r, theta)
        assert_almost_equal(
            ans, (ra2, dec2), err_msg="{0:5.2f},{1:5.2f} -> {2:g},{3:g} -> {4:5.2f},{5:5.2f} != {6:g},{7:g}".format(ra1, dec1, r, theta, ra2, dec2, *ans))


def test_dist_rhumb():
    """Test rhumb distance calculation"""
    for ra1, dec1, ra2, dec2, dist in [(0, 0, 0, 1, 1),
                                       (0, 0, 180, 0, 180)
                                       ]:
        ans = at.dist_rhumb(ra1, dec1, ra2, dec2)
        assert_almost_equal(ans, dist)


def test_bear_rhumb():
    """Test rhumb bearing calculation"""
    for ra1, dec1, ra2, dec2, bear in [(0, 0, 0, 1, 0),
                                       (0, 0, 180, 0, 90)
                                       ]:
        ans = at.bear_rhumb(ra1, dec1, ra2, dec2)
        assert_almost_equal(ans, bear)


def test_translate_rhumb():
    """Test translate along rhumb line"""
    for (ra1, dec1), (r, theta), (ra2, dec2) in [((0, 0), (1, 0), (0, 1)),
                                                 # negative r
                                                 ((12, -45), (-1, 180), (12, -44))
                                                 ]:
        ans = at.translate_rhumb(ra1, dec1, r, theta)
        assert_almost_equal(
            ans, (ra2, dec2), err_msg="{0:5.2f},{1:5.2f} -> {2:g},{3:g} -> {4:5.2f},{5:5.2f} != {6:g},{7:g}".format(ra1, dec1, r, theta, ra2, dec2, *ans))


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
