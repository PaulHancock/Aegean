#! /usr/bin/env python
"""
Test cluster.py
"""
import logging
import math
from copy import deepcopy

import numpy as np
from AegeanTools import catalogs, cluster, wcs_helpers
from AegeanTools.models import SimpleSource
from astropy.io import fits

__author__ = 'Paul Hancock'

logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)


def test_norm_dist():
    """Test norm_dist"""
    src1 = SimpleSource()
    src1.ra = 0
    src1.dec = 0
    src1.a = 1.
    src1.b = 1.
    src1.pa = 0.
    src2 = SimpleSource()
    src2.ra = 0
    src2.dec = 1/3600.
    src2.a = 1
    src2.b = 1
    src2.pa = 0.
    if not cluster.norm_dist(src1, src1) == 0:
        raise AssertionError()
    if not cluster.norm_dist(src1, src2) == 1/math.sqrt(2):
        raise AssertionError()


def test_sky_dist():
    """Test sky_dist"""
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


def test_vectorized():
    """Test that norm_dist and sky_dist can be vectorized"""
    # random data as struct array with interface like SimpleSource
    X = np.random.RandomState(0).rand(20, 6)
    Xr = np.rec.array(X.view([('ra', 'f8'), ('dec', 'f8'),
                              ('a', 'f8'), ('b', 'f8'),
                              ('pa', 'f8'),
                              ('peak_flux', 'f8')]).ravel())

    def to_ss(x):
        "Convert numpy.rec to SimpleSource"
        out = SimpleSource()
        for f in Xr.dtype.names:
            setattr(out, f, getattr(x, f))
        return out

    for dist in [cluster.norm_dist, cluster.sky_dist]:
        x0 = Xr[0]
        # calculate distance of x0 to all of Xr with vectorized operations:
        dx0all = dist(x0, Xr)
        for i, xi in enumerate(Xr):
            dx0xi = dist(x0, xi)
            # check equivalence between pairs of sources and vectorized
            if not np.isclose(dx0xi, dx0all[i], atol=0):
                raise AssertionError()
            # check equivalence between SimpleSource and numpy.record
            if not np.isclose(dx0xi, dist(to_ss(x0), to_ss(xi)), atol=0):
                raise AssertionError()


def test_pairwise_elliptical_binary():
    """Test pairwise_elliptical_binary distance"""
    src1 = SimpleSource()
    src1.ra = 0
    src1.dec = 0
    src1.a = 1.
    src1.b = 1.
    src1.pa = 0.
    src2 = deepcopy(src1)
    src2.dec = 1/3600.
    src3 = deepcopy(src1)
    src3.dec = 50
    mat = cluster.pairwise_ellpitical_binary([src1, src2, src3], eps=0.5)
    if not np.all(mat == [[False, True, False],
                          [True, False, False],
                          [False, False, False]]):
        raise AssertionError()


def test_regroup():
    """Test that regroup does things"""
    # this should throw an attribute error
    try:
        cluster.regroup([1], eps=1)
    except AttributeError as e:
        print(f"Correctly raised error {type(e)}")

    # this should result in 51 groups
    a = cluster.regroup('tests/test_files/1904_comp.fits',
                        eps=1/3600.)
    if not len(a) == 51:
        raise AssertionError(
            "Regroup with eps=1/3600. gave {0} groups instead of 51"
            .format(len(a)))

    # this should give 1 group
    a = cluster.regroup('tests/test_files/1904_comp.fits', eps=10, far=1000)
    if not len(a) == 1:
        raise AssertionError(
            "Regroup with eps=10, far=1000. gave {0} groups instead of 51"
            .format(len(a)))


def test_regroup_dbscan():
    table = catalogs.load_table('tests/test_files/1904_comp.fits')
    srccat = catalogs.table_to_source_list(table)
    a = cluster.regroup_dbscan(srccat,
                               eps=1/3600.)
    if not len(a) == 51:
        raise AssertionError(
            "Regroup_dbscan with eps=1/3600. gave {0} groups instead of 51"
            .format(len(a)))
    return


def test_resize_ratio():
    """Test that resize works with ratio"""
    # Load a table
    table = catalogs.load_table('tests/test_files/1904_comp.fits')
    srccat = catalogs.table_to_source_list(table)

    first = deepcopy(srccat[0])
    out = cluster.resize(deepcopy(srccat), ratio=1)

    if not ((first.a - out[0].a < 1e-9) and
            (first.b - out[0].b < 1e-9)):
        raise AssertionError("resize of 1 is not identity")

    return


def test_resize_psfhelper():
    """Test that resize works with psfhelpers"""
    # Load a table
    table = catalogs.load_table('tests/test_files/1904_comp.fits')
    srccat = catalogs.table_to_source_list(table)
    # make psfhelper
    head = fits.getheader('tests/test_files/1904-66_SIN.fits')
    psfhelper = wcs_helpers.WCSHelper.from_header(head)

    first = deepcopy(srccat[0])
    out = cluster.resize(deepcopy(srccat), psfhelper=psfhelper)
    print(first.a, out[0].a)

    if not ((first.a - out[0].a < 1e-9) and
            (first.b - out[0].b < 1e-9)):
        raise AssertionError("resize with psfhelper is not identity")

    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
