#! python
from __future__ import print_function
__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import angle_tools as at
from astropy.coordinates import Angle
import astropy.units as u
import numpy as np
from numpy.testing import assert_approx_equal, assert_almost_equal


def test_ra2dec():
    # Test against the astropy calculations
    for ra in ['14:21:45.003', '-12 04 22', '-00 01 12.003']:
        ans = at.ra2dec(ra)
        desired = Angle(ra, unit=u.hourangle).hour * 15
        assert_approx_equal(ans, desired, "{0} != {1}".format(ans, desired))


def test_dec2dec():
    # Test against the astropy calculations
    for dec in ['+14:21:45.003', '-99 04 22', '-00 01 23.456', '00 01']:
        ans = at.dec2dec(dec)
        desired = Angle(dec, unit=u.degree).degree
        assert_approx_equal(ans, desired, err_msg="{0} != {1}".format(ans, desired))


def test_dec2dms():
    for dec, dstr in [(-0.12345, "-00:07:24.42"),
                      (80.0, "+80:00:00.00"),
                      (np.nan, "XX:XX:XX.XX"),
                      (np.inf, "XX:XX:XX.XX")]:
        ans = at.dec2dms(dec)
        assert ans == dstr, "{0} != {1}".format(ans, dstr)


def test_dec2hms():
    for dec, dstr in [(-15, "23:00:00.00"),
                      (15, "01:00:00.00"),
                      (23.5678, "01:34:16.27"),
                      (np.nan, "XX:XX:XX.XX"),
                      (np.inf, "XX:XX:XX.XX")]:
        ans = at.dec2hms(dec)
        assert ans == dstr, "{0} != {1}".format(ans, dstr)


def test_gcd():
    for ra1, dec1, ra2, dec2, dist in [(0, 0, 0, 1, 1),  # simple 1 deg offset
                                       (0, -90, 180, 90, 180),  # pole to pole
                                       (120, 89, 300, 89, 2.),  # over the pole
                                       (0, 0, 179.99999, 0, 179.99999),  # distances very close to 180deg
                                       (12.0, -90, 45, -90, 0)  # at the south pole
                                       ]:
        ans = at.gcd(ra1, dec1, ra2, dec2)
        assert_almost_equal(ans, dist, err_msg="{0:5.2f},{1:5.2f} <-> {2:5.2f},{3:5.2f} == {4:g} != {5:g}".format(ra1, dec1, ra2, dec2, dist, ans))


def test_bear():
    for ra1, dec1, ra2, dec2, bear in [(0, 0, 0, 1, 0),
                                       (0, 0, 180, 90, 0),
                                       (0, 0, 179.99999, 0, 90),
                                       (0, 0, 180.00001, 0, -90)
                                       ]:
        ans = at.bear(ra1, dec1, ra2, dec2)
        assert_almost_equal(ans, bear, err_msg="{0:5.2f},{1:5.2f} <-> {2:5.2f},{3:5.2f} == {4:g} != {5:g}".format(ra1, dec1, ra2, dec2, bear, ans))


def test_translate():
    for (ra1, dec1), (r, theta), (ra2, dec2) in [((0, 0), (1, 0), (0, 1)),
                                                 ((45, 89.75), (0.5, 0), (225, 89.75)),  # over the pole
                                                 ((12, -45), (-1, 180), (12, -44))  # negative r
                                       ]:
        ans = at.translate(ra1, dec1, r, theta)
        assert_almost_equal(ans, (ra2, dec2), err_msg="{0:5.2f},{1:5.2f} -> {2:g},{3:g} -> {4:5.2f},{5:5.2f} != {6:g},{7:g}".format(ra1, dec1, r, theta, ra2, dec2, *ans))


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")