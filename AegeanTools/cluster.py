#! /usr/bin/env python

"""
Cluster and crossmatch tools and analysis functions.

Includes:
- DBSCAN clustering
"""

__author__= "Paul Hancock"

import numpy as np
import sys
import math

from angle_tools import gcd, bear
from catalogs import load_table, table_to_source_list

# join the Aegean logger
import logging
log = logging.getLogger('Aegean')

cc2fwhm = (2 * math.sqrt(2 * math.log(2)))
fwhm2cc = 1/cc2fwhm


def norm_dist(src1, src2):
    """
    Calculate the normalised distance between two sources.

    Sources are elliptical Gaussians.
    :param src1: type Aegean.models.SimpleSource
    :param src2: type Aegean.models.SimpleSource
    :return:
    """
    if src1 == src2:
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
    Return the distance between two sources in degrees

    A light wrapper around angle_tools.gcd
    """

    if src1 == src2:
        return 0
    return gcd(src1.ra, src1.dec, src2.ra, src2.dec) # degrees


def pairwise_ellpitical_binary(sources, eps, far=None):
    """
    Calculate the probability of an association between each pair of sources.
    0<= probability <=1
    Form this into a matrix.
    :param sources: A list of sources sorted by declination
    :return: a matrix of probabilities.
    """
    if far is None:
        far = max(a.a/3600 for a in sources)
    l = len(sources)
    distances = np.ones((l, l), dtype=bool)
    for i in xrange(l):
        for j in xrange(i,l):
            if j<i:
                continue
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


def regroup(catalog, eps, far=None, dist=norm_dist):
    """
    Regroup the islands of a catalog according to their normalised distance
    return a list of island groups, sources have their (island,source) parameters relabeled
    :param catalog: A list of objects with the following properties[units]:
                    ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]
    :param eps: maximum normalised distance within which sources are considered to be grouped
    :param far: (degrees) sources that are further than this distance appart will not be grouped, and will not be tested
    :param dist: a function that calculates the distance between two sources must accept two SimpleSource objects.
    :return: groups of sources as a dict {id:[src, ...], ...}
    """
    if isinstance(catalog, str):
        table = load_table(catalog)
        srccat = table_to_source_list(table)
    else:
        try:
            srccat = catalog
            _ = catalog[0].ra
            _ = catalog[0].dec
            _ = catalog[0].a
            _ = catalog[0].b
            _ = catalog[0].pa
            _ = catalog[0].peak_flux

        except AttributeError:
            log.error("catalog is not understood.")
            log.error("catalog: Should be a list of objects with the following properties[units]:\n" +
                      "ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]")
            sys.exit(1)

    log.info("Regrouping islands within catalog")
    log.debug("Calculating distances")

    # most negative declination first
    srccat = sorted(srccat, key = lambda x: x.dec)

    if far is None:
        far = 0.5 # 10*max(a.a/3600 for a in srccat)

    groups = {0: [srccat[0]]}
    last_group = 0

    # to parallelize this code, break the list into one part per core
    # compute the groups within each part
    # when the groups are found, check the last/first entry of pairs of groups to see if they need to be joined together
    for s1 in srccat[1:]:
        done = False
        # when an islands largest (last) declination is smaller than decmin, we don't need to look at any more islands
        decmin = s1.dec - far
        for g in xrange(last_group, -1, -1):
            if groups[g][-1].dec < decmin:
                break
            rafar = far / np.cos(np.radians(s1.dec))
            for s2 in groups[g]:
                if abs(s2.ra - s1.ra) > rafar:
                    continue
                if dist(s1, s2) < eps:
                    groups[g].append(s1)
                    done = True
                    break
            if done:
                break
        if not done:
            last_group += 1
            groups[last_group] = [s1]

    islands = []
    # now that we have the groups, we relabel the sources to have (island,component) in flux order
    # note that the order of sources within an island list is not changed - just their labels
    for isle in groups.keys():
        for comp, src in enumerate(sorted(groups[isle], key=lambda x: -1*x.peak_flux)):
            src.island = isle
            src.source = comp
        islands.append(groups[isle])
    return islands


def group_iter(catalog, eps, min_members=1):
    """
    :param catalog: List of sources, or filename of a catalog
    :param eps: Clustering radius in *degrees*
    :param min_members: Minimum number of members to form a cluster, default=1
    :yield: lists of sources, one list per group. No particular order.
    """
    import sklearn
    import sklearn.cluster

    if isinstance(catalog,str):
        table = load_table(catalog)
        srccat = table_to_source_list(table)
    elif isinstance(catalog,list):
        try:
            srccat = catalog
        except AttributeError:
            logging.error("Catalog is either not iterable, or its elements have not ra/dec attributes")
            sys.exit(1)
    else:
        logging.error("I don't know what catalog is")
        sys.exit(1)

    log.info("Regrouping islands within catalog")
    log.debug("Calculating distances")

    srccat = np.array(sorted(srccat, key = lambda x: x.dec))
    X = pairwise_ellpitical_binary(srccat,eps)

    log.debug("Clustering")
    samples, labels = sklearn.cluster.dbscan(X,eps=0.5, min_samples=min_members, metric='precomputed')
    # remove repeats and the noise flag of -1
    unique_labels = set(labels).difference(set([-1]))
    # Return groups of sources
    for l in unique_labels:
        class_member_mask = (labels == l)
        yield srccat[class_member_mask]


if __name__ == "__main__":
    logging.basicConfig()
    log = logging.getLogger('Aegean')
    catalog = '1904_comp.vot'
    catalog = 'GLEAM_IDR1.fits'
    table = load_table(catalog)
    positions = np.array(zip(table['ra'],table['dec']))
    srccat = list(table_to_source_list(table))
    # make the catalog stupid big for memory testing.
    #for i in xrange(5):
    #    srccat.extend(srccat)
    groups = regroup(srccat, eps=np.sqrt(2),far=0.277289506048)
    print "Sources ", len(table)
    print "Groups ", len(groups)
    for g in groups[:50]:
        print len(g),[(a.island,a.source) for a in g]
