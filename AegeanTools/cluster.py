#! /usr/bin/env python

"""
Cluster and crossmatch tools and analysis functions.

DBSCAN clustering
"""

__author__= "Paul Hancock"

import numpy as np
import logging
import sys

from angle_tools import gcd
import sklearn
import sklearn.cluster

from catalogs import load_table, table_to_source_list

def pairwise_distance(positions):
    """

    :param positions:
    :return:
    """
    distances = np.empty((len(positions), len(positions)), dtype=np.float32)
    for i,p1 in enumerate(positions):
        for j,p2 in enumerate(positions):
            if j<i: #distances are symmetric so only calculate 1/2 of them
                continue
            distances[i, j] = gcd(*np.ravel([p1,p2]))
            distances[j, i] = distances[i, j]
    return distances

def group_iter(catalog, eps, min_members=1):
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

    X = pairwise_distance(positions)
    samples, labels = sklearn.cluster.dbscan(X,eps=eps, min_samples=min_members, metric='precomputed')
    # remove repeats and the noise flag of -1
    unique_labels = set(labels).difference(set([-1]))
    # Return groups of sources
    for l in unique_labels:
        class_member_mask = (labels == l)
        yield srccat[class_member_mask]
