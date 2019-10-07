#! /usr/bin/env python

"""
Tools for manipulating angles on the surface of a sphere
- distance
- bearing between two points
- translation along a path
- paths are either great circles or rhumb lines

also angle <-> string conversion tools for Aegean
"""

__author__ = "Paul Hancock"

import math
import numpy as np


def ra2dec(ra):
    """
    Convert sexegessimal RA string into a float in degrees.

    Parameters
    ----------
    ra : str
        A string separated representing the RA.
        Expected format is `hh:mm[:ss.s]`
        Colons can be replaced with any whit space character.

    Returns
    -------
    ra : float
        The RA in degrees.
    """
    return dec2dec(ra)*15


def dec2dec(dec):
    """
    Convert sexegessimal RA string into a float in degrees.

    Parameters
    ----------
    dec : str
        A string separated representing the Dec.
        Expected format is `[+- ]hh:mm[:ss.s]`
        Colons can be replaced with any whit space character.

    Returns
    -------
    dec : float
        The Dec in degrees.
    """
    d = dec.replace(':', ' ').split()
    if len(d) == 2:
        d.append('0.0')
    if d[0].startswith('-') or float(d[0]) < 0:
        return float(d[0]) - float(d[1]) / 60.0 - float(d[2]) / 3600.0
    return float(d[0]) + float(d[1]) / 60.0 + float(d[2]) / 3600.0


def dec2dms(x):
    """
    Convert decimal degrees into a sexagessimal string in degrees.

    Parameters
    ----------
    x : float
        Angle in degrees

    Returns
    -------
    dms : str
        String of format [+-]DD:MM:SS.SS
        or XX:XX:XX.XX if x is not finite.
    """
    if not np.isfinite(x):
        return 'XX:XX:XX.XX'
    if x < 0:
        sign = '-'
    else:
        sign = '+'
    x = abs(x)
    d = int(math.floor(x))
    m = int(math.floor((x - d) * 60))
    s = float(( (x - d) * 60 - m) * 60)
    return '{0}{1:02d}:{2:02d}:{3:05.2f}'.format(sign, d, m, s)


def dec2hms(x):
    """
    Convert decimal degrees into a sexagessimal string in hours.

    Parameters
    ----------
    x : float
        Angle in degrees

    Returns
    -------
    dms : string
        String of format HH:MM:SS.SS
        or XX:XX:XX.XX if x is not finite.
    """
    if not np.isfinite(x):
        return 'XX:XX:XX.XX'
    # wrap negative RA's
    if x < 0:
        x += 360
    x /= 15.0
    h = int(x)
    x = (x - h) * 60
    m = int(x)
    s = (x - m) * 60
    return '{0:02d}:{1:02d}:{2:05.2f}'.format(h, m, s)


# The following functions are explained at http://www.movable-type.co.uk/scripts/latlong.html
# phi ~ lat ~ Dec
# lambda ~ lon ~ RA
def gcd(ra1, dec1, ra2, dec2):
    """
    Calculate the great circle distance between to points using the haversine formula [1]_.


    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The coordinates of the two points of interest.
        Units are in degrees.

    Returns
    -------
    dist : float
        The distance between the two points in degrees.

    Notes
    -----
    This duplicates the functionality of astropy but is faster as there is no creation of SkyCoords objects.

    .. [1] `Haversine formula <https://en.wikipedia.org/wiki/Haversine_formula>`_
    """
    # TODO:  Vincenty formula see - https://en.wikipedia.org/wiki/Great-circle_distance
    dlon = ra2 - ra1
    dlat = dec2 - dec1
    a = np.sin(np.radians(dlat) / 2) ** 2
    a += np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.sin(np.radians(dlon) / 2) ** 2
    sep = np.degrees(2 * np.arcsin(np.minimum(1, np.sqrt(a))))
    return sep


def bear(ra1, dec1, ra2, dec2):
    """
    Calculate the bearing of point 2 from point 1 along a great circle.
    The bearing is East of North and is in [0, 360), whereas position angle is also East of North but (-180,180]

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The sky coordinates (degrees) of the two points.

    Returns
    -------
    bear : float
        The bearing of point 2 from point 1 (degrees).
    """
    rdec1 = np.radians(dec1)
    rdec2 = np.radians(dec2)
    rdlon = np.radians(ra2-ra1)
    y = np.sin(rdlon) * np.cos(rdec2)
    x = np.cos(rdec1) * np.sin(rdec2)
    x -= np.sin(rdec1) * np.cos(rdec2) * np.cos(rdlon)
    return np.degrees(np.arctan2(y, x))


def translate(ra, dec, r, theta):
    """
    Translate a given point a distance r in the (initial) direction theta, along a  great circle.


    Parameters
    ----------
    ra, dec : float
        The initial point of interest (degrees).
    r, theta : float
        The distance and initial direction to translate (degrees).

    Returns
    -------
    ra, dec : (float, float)
        The translated position (degrees).
    """
    factor = np.sin(np.radians(dec)) * np.cos(np.radians(r))
    factor += np.cos(np.radians(dec)) * np.sin(np.radians(r)) * np.cos(np.radians(theta))
    dec_out = np.degrees(np.arcsin(factor))

    y = np.sin(np.radians(theta)) * np.sin(np.radians(r)) * np.cos(np.radians(dec))
    x = np.cos(np.radians(r)) - np.sin(np.radians(dec)) * np.sin(np.radians(dec_out))
    ra_out = ra + np.degrees(np.arctan2(y, x))
    return ra_out, dec_out


def dist_rhumb(ra1, dec1, ra2, dec2):
    """
    Calculate the Rhumb line distance between two points [1]_.
    A Rhumb line between two points is one which follows a constant bearing.

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The position of the two points (degrees).

    Returns
    -------
    dist : float
        The distance between the two points along a line of constant bearing.

    Notes
    -----
    .. [1] `Rhumb line <https://en.wikipedia.org/wiki/Rhumb_line>`_
    """
    # verified against website to give correct results
    phi1 = np.radians(dec1)
    phi2 = np.radians(dec2)
    dphi = phi2 - phi1
    lambda1 = np.radians(ra1)
    lambda2 = np.radians(ra2)
    dpsi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    if dpsi < 1e-12:
        q = np.cos(phi1)
    else:
        q = dpsi / dphi
    dlambda = lambda2 - lambda1
    if dlambda > np.pi:
        dlambda -= 2 * np.pi
    dist = np.hypot(dphi, q * dlambda)
    return np.degrees(dist)


def bear_rhumb(ra1, dec1, ra2, dec2):
    """
    Calculate the bearing of point 2 from point 1 along a Rhumb line.
    The bearing is East of North and is in [0, 360), whereas position angle is also East of North but (-180,180]

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        The sky coordinates (degrees) of the two points.

    Returns
    -------
    dist : float
        The bearing of point 2 from point 1 along a Rhumb line (degrees).
    """
    # verified against website to give correct results
    phi1 = np.radians(dec1)
    phi2 = np.radians(dec2)
    lambda1 = np.radians(ra1)
    lambda2 = np.radians(ra2)
    dlambda = lambda2 - lambda1

    dpsi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))

    theta = np.arctan2(dlambda, dpsi)
    return np.degrees(theta)


def translate_rhumb(ra, dec, r, theta):
    """
    Translate a given point a distance r in the (initial) direction theta, along a Rhumb line.

    Parameters
    ----------
    ra, dec : float
        The initial point of interest (degrees).
    r, theta : float
        The distance and initial direction to translate (degrees).

    Returns
    -------
    ra, dec : float
        The translated position (degrees).
    """
    # verified against website to give correct results
    # with the help of http://williams.best.vwh.net/avform.htm#Rhumb
    delta = np.radians(r)
    phi1 = np.radians(dec)
    phi2 = phi1 + delta * np.cos(np.radians(theta))
    dphi = phi2 - phi1

    if abs(dphi) < 1e-9:
        q = np.cos(phi1)
    else:
        dpsi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
        q = dphi / dpsi

    lambda1 = np.radians(ra)
    dlambda = delta * np.sin(np.radians(theta)) / q
    lambda2 = lambda1 + dlambda

    ra_out = np.degrees(lambda2)
    dec_out = np.degrees(phi2)
    return ra_out, dec_out
