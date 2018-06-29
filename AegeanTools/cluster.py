#! /usr/bin/env python
"""
Cluster and crossmatch tools and analysis functions.

Includes:
- DBSCAN clustering
"""

from __future__ import print_function

__author__= "Paul Hancock"

import numpy as np
import math

from .angle_tools import gcd, bear
from .catalogs import load_table, table_to_source_list

# join the Aegean logger
import logging
log = logging.getLogger('Aegean')

cc2fwhm = (2 * math.sqrt(2 * math.log(2)))
fwhm2cc = 1/cc2fwhm


def norm_dist(src1, src2):
    """
    Calculate the normalised distance between two sources.
    Sources are elliptical Gaussians.

    The normalised distance is calculated as the GCD distance between the centers,
    divided by quadrature sum of the radius of each ellipse along a line joining the two ellipses.

    For ellipses that touch at a single point, the normalized distance will be 1/sqrt(2).

    Parameters
    ----------
    src1, src2 : object
        The two positions to compare. Objects must have the following parameters: (ra, dec, a, b, pa).

    Returns
    -------
    dist: float
        The normalised distance.

    """
    if np.all(src1 == src2):
        return 0
    dist = gcd(src1.ra, src1.dec, src2.ra, src2.dec) # degrees

    # the angle between the ellipse centers
    phi = bear(src1.ra, src1.dec, src2.ra, src2.dec) # Degrees
    # Calculate the radius of each ellipse along a line that joins their centers.
    r1 = src1.a*src1.b / np.hypot(src1.a * np.sin(np.radians(phi - src1.pa)),
                                  src1.b * np.cos(np.radians(phi - src1.pa)))
    r2 = src2.a*src2.b / np.hypot(src2.a * np.sin(np.radians(180 + phi - src2.pa)),
                                  src2.b * np.cos(np.radians(180 + phi - src2.pa)))
    R = dist / (np.hypot(r1, r2) / 3600)
    return R


def sky_dist(src1, src2):
    """
    Great circle distance between two sources.
    A check is made to determine if the two sources are the same object, in this case
    the distance is zero.

    Parameters
    ----------
    src1, src2 : object
        Two sources to check. Objects must have parameters (ra,dec) in degrees.

    Returns
    -------
    distance : float
        The distance between the two sources.

    See Also
    --------
    :func:`AegeanTools.angle_tools.gcd`
    """

    if np.all(src1 == src2):
        return 0
    return gcd(src1.ra, src1.dec, src2.ra, src2.dec) # degrees


def pairwise_ellpitical_binary(sources, eps, far=None):
    """
    Do a pairwise comparison of all sources and determine if they have a normalized distance within
    eps.

    Form this into a matrix of shape NxN.


    Parameters
    ----------
    sources : list
        A list of sources (objects with parameters: ra,dec,a,b,pa)

    eps : float
        Normalised distance constraint.

    far : float
        If sources have a dec that differs by more than this amount then they are considered to be not matched.
        This is a short-cut around performing GCD calculations.

    Returns
    -------
    prob : numpy.ndarray
        A 2d array of True/False.

    See Also
    --------
    :func:`AegeanTools.cluster.norm_dist`
    """
    if far is None:
        far = max(a.a/3600 for a in sources)
    l = len(sources)
    distances = np.zeros((l, l), dtype=bool)
    for i in range(l):
        for j in range(i, l):
            if i == j:
                distances[i, j] = False
                continue
            src1 = sources[i]
            src2 = sources[j]
            if src2.dec - src1.dec > far:
                break
            if abs(src2.ra - src1.ra)*np.cos(np.radians(src1.dec)) > far:
                continue
            distances[i, j] = norm_dist(src1, src2) > eps
            distances[j, i] = distances[i, j]
    return distances


def regroup_vectorized(srccat, eps, far=None, dist=norm_dist):
    """
    Regroup the islands of a catalog according to their normalised distance.

    Assumes srccat is recarray-like for efficiency.
    Return a list of island groups.

    Parameters
    ----------
    srccat : np.rec.arry or pd.DataFrame
        Should have the following fields[units]:
        ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]
    eps : float
        maximum normalised distance within which sources are considered to be
        grouped
    far : float
        (degrees) sources that are further than this distance apart will not
        be grouped, and will not be tested.
        Default = 0.5.
    dist : func
        a function that calculates the distance between a source and each
        element of an array of sources.
        Default = :func:`AegeanTools.cluster.norm_dist`

    Returns
    -------
    islands : list of lists
        Each island contians integer indices for members from srccat
        (in descending dec order).
    """
    if far is None:
        far = 0.5  # 10*max(a.a/3600 for a in srccat)

    # most negative declination first
    # XXX: kind='mergesort' ensures stable sorting for determinism.
    #      Do we need this?
    order = np.argsort(srccat.dec, kind='mergesort')[::-1]
    # TODO: is it better to store groups as arrays even if appends are more
    #       costly?
    groups = [[order[0]]]
    for idx in order[1:]:
        rec = srccat[idx]
        # TODO: Find out if groups are big enough for this to give us a speed
        #       gain. If not, get distance to all entries in groups above
        #       decmin simultaneously.
        decmin = rec.dec - far
        for group in reversed(groups):
            # when an island's largest (last) declination is smaller than
            # decmin, we don't need to look at any more islands
            if srccat.dec[group[-1]] < decmin:
                # new group
                groups.append([idx])
            rafar = far / np.cos(np.radians(rec.dec))
            group_recs = np.take(srccat, group, mode='clip')
            group_recs = group_recs[abs(rec.ra - group_recs.ra) <= rafar]
            if len(group_recs) and dist(rec, group_recs).min() < eps:
                group.append(idx)
                break
        else:
            # new group
            groups.append([idx])

    # TODO?: a more numpy-like interface would return only an array providing
    #        the mapping:
    #    group_idx = np.empty(len(srccat), dtype=int)
    #    for i, group in enumerate(groups):
    #        group_idx[group] = i
    #    return group_idx
    return groups


def regroup(catalog, eps, far=None, dist=norm_dist):
    """
    Regroup the islands of a catalog according to their normalised distance.
    Return a list of island groups. Sources have their (island,source) parameters relabeled.


    Parameters
    ----------
    catalog : str or object
        Either a filename to read into a source list, or a list of objects with the following properties[units]:
        ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]

    eps : float
        maximum normalised distance within which sources are considered to be grouped

    far : float
        (degrees) sources that are further than this distance appart will not be grouped, and will not be tested.
        Default = None.

    dist : func
        a function that calculates the distance between two sources must accept two SimpleSource objects.
        Default = :func:`AegeanTools.cluster.norm_dist`

    Returns
    -------
    islands : list
        A list of islands. Each island is a list of sources.

    See Also
    --------
    :func:`AegeanTools.cluster.norm_dist`
    """

    if isinstance(catalog, str):
        table = load_table(catalog)
        srccat = table_to_source_list(table)
    else:
        try:
            srccat = catalog
            _ = catalog[0].ra, catalog[0].dec, catalog[0].a, catalog[0].b, catalog[0].pa, catalog[0].peak_flux

        except AttributeError as e:
            log.error("catalog is not understood.")
            log.error("catalog: Should be a list of objects with the following properties[units]:\n" +
                      "ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]")
            raise e

    log.info("Regrouping islands within catalog")
    log.debug("Calculating distances")

    if far is None:
        far = 0.5  # 10*max(a.a/3600 for a in srccat)

    srccat_array = np.rec.fromrecords(
        [(s.ra, s.dec, s.a, s.b, s.pa, s.peak_flux)
         for s in srccat],
        names=['ra', 'dec', 'a', 'b', 'pa', 'peak_flux'])
    groups = regroup_vectorized(srccat_array, eps=eps, far=far, dist=dist)
    groups = [[srccat[idx] for idx in group]
              for group in groups]

    islands = []
    # now that we have the groups, we relabel the sources to have (island,component) in flux order
    # note that the order of sources within an island list is not changed - just their labels
    for isle, group in enumerate(groups):
        for comp, src in enumerate(sorted(group, key=lambda x: -1*x.peak_flux)):
            src.island = isle
            src.source = comp
        islands.append(group)
    return islands
