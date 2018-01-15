#! python
from __future__ import print_function

__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import cluster
from AegeanTools.models import SimpleSource
from copy import deepcopy, copy
import math
import numpy as np

import logging
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)


def test_norm_dist():
    src1 = SimpleSource()
    src1.ra = 0
    src1.dec = 0
    src1.a = 1.
    src1.b = 1.
    src2 = SimpleSource()
    src2.ra = 0
    src2.dec = 1/3600.
    src2.a = 1
    src2.b = 1
    if not cluster.norm_dist(src1, src1) == 0:
        raise AssertionError()
    if not cluster.norm_dist(src1, src2) == 1/math.sqrt(2):
        raise AssertionError()


def test_sky_dist():
    src1 = SimpleSource()
    src1.ra = 0
    src1.dec = 0
    src2 = SimpleSource()
    src2.ra = 0
    src2.dec = 1/3600.
    if not cluster.sky_dist(src1, src1) == 0.:
        raise AssertionError()
    if not cluster.sky_dist(src1, src2) == 1/3600.:
        raise AssertionError()


def test_pairwise_elliptical_binary():
    src1 = SimpleSource()
    src1.ra = 0
    src1.dec = 0
    src1.a = 1.
    src1.b = 1.
    src2 = deepcopy(src1)
    src2.dec = 1/3600.
    src3 = deepcopy(src1)
    src3.dec = 50
    mat = cluster.pairwise_ellpitical_binary([src1, src2, src3], eps=0.5)
    if not np.all(mat == [[False, True, False], [True, False, False], [False, False, False]]):
        raise AssertionError()


def test_regroup():
    # this should throw an attribute error
    try:
        cluster.regroup([1], eps=1)
    except AttributeError as e:
        pass

    # this should result in 51 groups
    a = cluster.regroup('tests/test_files/1904_comp.fits', eps=1/3600.)
    if not len(a) == 51:
        raise AssertionError()

    # this should give 1 group
    a = cluster.regroup('tests/test_files/1904_comp.fits', eps=10, far=1000)
    if not len(a) == 1:
        raise AssertionError()


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()