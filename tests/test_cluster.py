#! python
from __future__ import print_function

__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import cluster
from AegeanTools.models import SimpleSource
from copy import deepcopy, copy
import math
import numpy as np


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
    assert cluster.norm_dist(src1, src1) == 0
    assert cluster.norm_dist(src1, src2) == 1/math.sqrt(2)


def test_sky_dist():
    src1 = SimpleSource()
    src1.ra = 0
    src1.dec = 0
    src2 = SimpleSource()
    src2.ra = 0
    src2.dec = 1/3600.
    assert cluster.sky_dist(src1, src1) == 0.
    assert cluster.sky_dist(src1, src2) == 1/3600.


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
    assert np.all(mat == [[False, True, False], [True, False, False], [False, False, False]])


def test_regroup():
    # this should throw an attribute error
    try:
        cluster.regroup([1], eps=1)
    except AttributeError as e:
        pass

    # this should result in 51 groups
    a = cluster.regroup('tests/test_files/1904_comp.fits', eps=1/3600.)
    assert len(a) == 51
    # this should give 1 group
    a = cluster.regroup('tests/test_files/1904_comp.fits', eps=10, far=1000)
    assert len(a) == 1

if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")