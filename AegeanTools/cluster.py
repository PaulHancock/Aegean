#! /usr/bin/env python

"""
Cluster and crossmatch tools and analysis functions.

DBSCAN clustering
"""

__author__= "Paul Hancock"

import numpy as np
import sys
import math

from angle_tools import gcd, bear
import sklearn
import sklearn.cluster

from catalogs import load_table, table_to_source_list

# join the Aegean logger
import logging
log = logging.getLogger('Aegean')

cc2fwhm = (2 * math.sqrt(2 * math.log(2)))
fwhm2cc = 1/cc2fwhm

def norm_dist(src1,src2):
    """
    Calculate the normalised distance between two sources.
    Sources are elliptical gaussians.
    :param src1:
    :param src2:
    :return:
    """
    if src1 == src2:
        return 0
    dist = gcd(src1.ra, src1.dec, src2.ra, src2.dec) # degrees

    # the angle between the ellipse centers
    phi = bear(src1.ra, src1.dec, src2.ra, src2.dec) # Degrees
    # Calculate the radius of each ellpise along a line that joins their centers.
    r1 = src1.a*src1.b / np.hypot(src1.a * np.sin(np.radians(phi - src1.pa)),
                                  src1.b * np.cos(np.radians(phi - src1.pa)))
    r2 = src2.a*src2.b / np.hypot(src2.a * np.sin(np.radians(180 + phi - src2.pa)),
                                  src2.b * np.cos(np.radians(180 + phi - src2.pa)))
    R = dist / (np.hypot(r1,r2) / 3600)
    return R

# import line_profiler
#@profile
def pairwise_ellpitical(sources, far = None):
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
    distances = np.full((l, l), fill_value = 50, dtype=np.float32)
    for i in xrange(l):
        for j in xrange(i,l):
            if j<i:
                continue
            if i == j:
                distances[i, j] = 0
                continue
            src1 = sources[i]
            src2 = sources[j]
            if src2.dec - src1.dec > far:
                break
            if abs(src2.ra - src1.ra)*np.cos(np.radians(src1.dec)) > far:
                continue
            distances[i, j] = norm_dist(src1, src2)
            distances[j, i] = distances[i, j]
    return distances

#@profile
def pairwise_ellpitical_binary(sources, eps, far = None):
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


def pairwise_distance(positions):
    """
    Calculate the distance between each pair of positions.
    Form this into a matrix.
    :param positions: A list of (ra,dec) positions in degrees
    :return: a matrix of distances between each pair of points.
    """
    distances = np.empty((len(positions), len(positions)), dtype=np.float32)
    for i,p1 in enumerate(positions):
        for j,p2 in enumerate(positions):
            if j<i: #distances are symmetric so only calculate 1/2 of them
                continue
            distances[i, j] = gcd(*np.ravel([p1,p2]))
            distances[j, i] = distances[i, j]
    return distances


def group_iter_dep(catalog, eps, min_members=1):
    """
    :param catalog: List of sources, or filename of a catalog
    :param eps: Clustering radius in *degrees*
    :param min_members: Minimum number of members to form a cluster, default=1
    :yield: lists of sources, one list per group. No particular order.
    """

    if isinstance(catalog,str):
        table = load_table(catalog)
        srccat = table_to_source_list(table)
    elif isinstance(catalog,list):
        try:
            srccat = catalog
        except AttributeError:
            logging.error("catalog is as list of something that has not ra/dec attributes")
            sys.exit(1)
    else:
        logging.error("I don't know what catalog is")
        sys.exit(1)

    log.info("Regrouping islands within catalog")
    log.debug("Calculating distances")

    srccat = np.array(sorted(srccat, key = lambda x: x.dec))
    X = pairwise_ellpitical(srccat)

    log.debug("Clustering")
    off_diag = X[np.where(X>0)]
    if min(off_diag) > eps:
        unique_labels = np.arange(len(srccat))
        labels = unique_labels
        log.info("None of the sources are clustered.")
    elif max(off_diag) < eps:
        unique_labels = [1]
        labels = unique_labels
        log.warn("All of the sources are within the same cluster")
    else:
        samples, labels = sklearn.cluster.dbscan(X,eps=eps, min_samples=min_members, metric='precomputed')
        # remove repeats and the noise flag of -1
        unique_labels = set(labels).difference(set([-1]))
    # Return groups of sources
    for l in unique_labels:
        class_member_mask = (labels == l)
        yield srccat[class_member_mask]

def group_iter(catalog, eps, min_members=1):
    """
    :param catalog: List of sources, or filename of a catalog
    :param eps: Clustering radius in *degrees*
    :param min_members: Minimum number of members to form a cluster, default=1
    :yield: lists of sources, one list per group. No particular order.
    """

    if isinstance(catalog,str):
        table = load_table(catalog)
        srccat = table_to_source_list(table)
    elif isinstance(catalog,list):
        try:
            srccat = catalog
        except AttributeError:
            logging.error("catalog is as list of something that has no ra/dec attributes")
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
    table = load_table(catalog)
    positions = np.array(zip(table['ra'],table['dec']))
    srccat = list(table_to_source_list(table))
    # make the catalog stupid big for memory testing.
    for i in xrange(5):
        srccat.extend(srccat)
    #X = pairwise_ellpitical_binary(srccat, eps=np.sqrt(2))
    #print X[:10,:10]

    for g in group_iter_binary(srccat, eps=np.sqrt(2)):
        print len(g),[(a.island,a.source) for a in g]
    for g in group_iter(srccat, eps = np.sqrt(2)):
        print len(g),[(a.island,a.source) for a in g]
