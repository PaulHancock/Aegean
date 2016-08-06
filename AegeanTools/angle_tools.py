#! /usr/bin/env python

"""
tools for manipulating angles on the surface of a sphere
- distance
- bearing between two points
- translation along a path
- paths are either great circles or rhumb lines

also angle <-> string conversion tools for Aegean
Will eventually be replaced with those from Astropy
"""

__author__ = "Paul Hancock"

import math
import numpy as np


def ra2dec(ra):
    """
    Accepts a string right ascention and converts it to decimal degrees
    requires hh:mm[:ss.s]
    """
    r = ra.replace(':', ' ').split()
    if len(r) == 2:
        r.append(0.0)
    return (float(r[0]) + float(r[1]) / 60.0 + float(r[2]) / 3600.0) * 15


def dec2dec(dec):
    """
    Accepts a string declination and converts it to decimal degrees
    requires +/-dd:mm[:ss.s]
    """
    d = dec.split(':')
    if len(d) == 2:
        d.append(0.0)
    if d[0].startswith('-') or float(d[0]) < 0:
        return float(d[0]) - float(d[1]) / 60.0 - float(d[2]) / 3600.0
    return float(d[0]) + float(d[1]) / 60.0 + float(d[2]) / 3600.0


def dec2dms(x):
    """
    Convert decimal degrees into a sexagessimal DD:MM:SS.SS string
    :param x:
    :return:
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
    Convert decimal degrees into a sexagessimal HH:MM:SS.SS string
    :param x:
    :return:
    """
    if not np.isfinite(x):
        return 'XX:XX:XX.XX'
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
    Great circle distance as calculated by the haversine formula
    ra/dec in degrees
    returns:
    sep in degrees
    """
    # TODO:  Vincenty formula see - https://en.wikipedia.org/wiki/Great-circle_distance
    dlon = ra2 - ra1
    dlat = dec2 - dec1
    a = np.sin(np.radians(dlat) / 2) ** 2
    a += np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.sin(np.radians(dlon) / 2) ** 2
    sep = np.degrees(2 * np.arcsin(min(1, np.sqrt(a))))
    return sep


def bear(ra1, dec1, ra2, dec2):
    """
    Calculate the bearing of point b from point a.
    bearing is East of North [0,360)
    position angle is East of North (-180,180]
    """
    dlon = ra2 - ra1
    # dlat = dec2 - dec1
    y = np.sin(np.radians(dlon)) * np.cos(np.radians(dec2))
    x = np.cos(np.radians(dec1)) * np.sin(np.radians(dec2))
    x -= np.sin(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.cos(np.radians(dlon))
    return np.degrees(np.arctan2(y, x))


def translate(ra, dec, r, theta):
    """
    Translate the point (ra,dec) a distance r (degrees) along angle theta (degrees)
    The translation is taken along an arc of a great circle.
    Return the (ra,dec) of the translated point.
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
    Rhumb line distance between two points
    distance is in degrees
    :param ra1:
    :param dec1:
    :param ra2:
    :param dec1:
    :return:
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
    The true bearing of a rhumb line that joins to points
    return bearing in degrees
    :param ra1:
    :param dec1:
    :param ra2:
    :param dec2:
    :return:
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
    Translate the point (ra,dec) a distance r (degrees) along angle theta (degrees)
    The translation is taken along an arc of a rhumb line.
    Return the (ra,dec) of the translated point.
    :param ra:
    :param dec:
    :param r:
    :param theta:
    :return:
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
