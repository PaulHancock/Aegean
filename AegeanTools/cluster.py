#! /usr/bin/env python

"""
Cluster and crossmatch tools and analysis functions.

DBSCAN clustering
"""

__author__= "Paul Hancock"

import numpy as np
import sys
import math

from angle_tools import gcd
import sklearn
import sklearn.cluster

from catalogs import load_table, table_to_source_list
from fitting import elliptical_gaussian

# join the Aegean logger
import logging
log = logging.getLogger('Aegean')

cc2fwhm = (2 * math.sqrt(2 * math.log(2)))
fwhm2cc = 1/cc2fwhm

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


def fa2cd(ab):
    """
    Convert fwhm in arcseconds to sigmas in degrees
    :param ab:
    :return:
    """
    return ab/3600.*fwhm2cc

def prob_elliptical(src1,src2):
    if src1 == src2:
        return 0
    #print src1.ra,src1.dec,fa2cd(src1.a),fa2cd(src1.b),-1*np.radians(src1.pa)
    #print src2.ra,src2.dec,fa2cd(src2.a),fa2cd(src2.b),-1*np.radians(src2.pa)
    p1 = elliptical_gaussian(src2.ra,src2.dec,1.,src1.ra,src1.dec,fa2cd(src1.a),fa2cd(src1.b),-1*np.radians(src1.pa))
    p2 = elliptical_gaussian(src1.ra,src1.dec,1.,src2.ra,src2.dec,fa2cd(src2.a),fa2cd(src2.b),-1*np.radians(src2.pa))
    print p1, p2
    return 1- np.sqrt(p1*p2)


def pairwise_probability(sources):
    """
    Calculate the probability of an association between each pair of sources.
    0<= probability <=1
    Form this into a matrix.
    :param sources: A list of sources
    :return: a matrix of probabilities.
    """
    probabilities = np.empty((len(sources), len(sources)), dtype=np.float32)
    for i,src1 in enumerate(sources):
        for j,src2 in enumerate(sources):
            if j<i:
                continue
            probabilities[i, j] = prob_elliptical(src1, src2)
            probabilities[j, i] = probabilities[i, j]
    return probabilities


def group_iter(catalog, eps, min_members=1, prob=False):
    """
    :param catalog: List of sources, or filename of a catalog
    :param eps: Clustering radius in *degrees*
    :param min_members: Minimum number of members to form a cluster, default=1
    :yiled: lists of sources, one list per group. No particular order.
    """

    if isinstance(catalog,str):
        table = load_table(catalog)
        positions = np.array(zip(table['ra'],table['dec']))
        srccat = np.array(table_to_source_list(table))
    elif isinstance(catalog,list):
        try:
            positions = np.array([(s.ra,s.dec) for s in catalog])
            srccat = np.array(catalog)
        except AttributeError:
            logging.error("catalog is as list of something that has not ra/dec attributes")
            sys.exit(1)
    else:
        logging.error("I don't know what catalog is")
        sys.exit(1)

    log.debug("Calculating distances")
    if prob:
        X = pairwise_probability(srccat)
    else:
        X = pairwise_distance(positions)
    log.debug("Clustering")
    samples, labels = sklearn.cluster.dbscan(X,eps=eps, min_samples=min_members, metric='precomputed')
    # remove repeats and the noise flag of -1
    unique_labels = set(labels).difference(set([-1]))
    # Return groups of sources
    for l in unique_labels:
        class_member_mask = (labels == l)
        yield srccat[class_member_mask]


if __name__ == "__main__":
    catalog = 'Test/Catalogs/1904_comp.vot'
    table = load_table(catalog)
    positions = np.array(zip(table['ra'],table['dec']))
    srccat = list(table_to_source_list(table))
    for g in group_iter(srccat,0.999,prob=True):
        print len(g),[(a.island,a.source) for a in g]
