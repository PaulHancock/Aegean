#! /usr/bin/env python
"""
Test models.py
"""

__author__ = 'Paul Hancock'

from AegeanTools import models
import numpy as np


def test_simple_source():
    """Make a new source without failing"""
    # make a new source without failing
    ss = models.SimpleSource()
    ss.ra = np.float32(12)
    ss.dec = ss.peak_flux = ss.err_peak_flux = ss.a = ss.b = ss.pa = 0.
    ss.local_rms = ss.background = ss.peak_pixel = 0.
    ss._sanitise()
    if not (isinstance(ss.ra, np.float64)): raise AssertionError()
    # convert to string without failing
    a = "{0}".format(ss)
    if not (a == ' 12.0000000   0.0000000  0.000000  0.000000  0.00  0.00    0.0 0000000'): raise AssertionError()
    if not (ss.__repr__() == ss.__str__()): raise AssertionError()
    if not (np.all(ss.as_list()[:-1] == [0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0])): raise AssertionError()
    isl = models.IslandSource()
    isl2 = models.IslandSource()
    if not (isl < ss): raise AssertionError()
    if not (isl <= ss): raise AssertionError()
    if isl > ss: raise AssertionError()
    if isl >= ss: raise AssertionError()
    if not (isl == isl2): raise AssertionError()
    if isl != isl2: raise AssertionError()
    out = models.ComponentSource()
    out.source = 1
    out2 = models.ComponentSource()
    out2.source = 2
    out3 = models.ComponentSource()
    out3.island = 1
    if not (out < out2): raise AssertionError()
    if not (out3 > out2): raise AssertionError()
    if not (out <= out2): raise AssertionError()
    if not (out3 >= out): raise AssertionError()
    if not (out != out2): raise AssertionError()
    if not (out == out): raise AssertionError()


def test_global_fitting_data():
    """Test that GlobalFittingData doesn't crash"""
    models.GlobalFittingData()


def test_island_fitting_data():
    """Test that IslandFittingData doesn't crash"""
    models.IslandFittingData()


def test_classify_catalogue():
    """Test classify_catalogue"""
    ss = []
    isl = []
    out = []
    mixed = []
    for i in range(10):
        a = models.SimpleSource()
        ss.append(a)
        b = models.IslandSource()
        b.island = i
        isl.append(b)
        c = models.ComponentSource()
        c.island = i
        d = models.ComponentSource()
        d.island = i
        d.source = 1
        out.extend([c, d])
        mixed.extend([a, b, c, d])
    a, b, c = models.classify_catalog(mixed)
    if not (np.all(b == isl)): raise AssertionError()
    if not (np.all(a == out)): raise AssertionError()
    groups = list(models.island_itergen(a))
    if not (len(groups) == 10): raise AssertionError()


def test_PixelIsland():
    """Tests"""
    pi = models.PixelIsland()

    # should complain about 3d data, when default is dim=2
    try:
        pi.set_mask(np.ones((2,2,2)))
    except AssertionError:
        pass
    else:
        raise AssertionError("set_mask should complain when given 3d data")

    data = np.ones((2,2))
    pi.set_mask(data)
    if pi.mask is not data:
        raise AssertionError("set_mask is not storing the mask properly")

    try:
        pi.calc_bounding_box(data, offsets=[0,0,0])
    except AssertionError:
        pass
    else:
        raise AssertionError("calc_bounding_box should complain about mismatched offsets")

    pi.calc_bounding_box(data, offsets=[0,0])
    if pi.mask is not data:
        raise AssertionError("calc_bounding_box is not storing the mask properly")

    if not np.all(pi.bounding_box == [[0,2],[0,2]]):
        raise AssertionError("bounding box not computed correctly")

    data = np.zeros((5,5))
    data[2, 3] = 1
    pi.calc_bounding_box(data, offsets=[0,0])
    if not np.all(pi.bounding_box == [[2,3],[3,4]]):
        print(pi.mask)
        print(pi.bounding_box)
        raise AssertionError("bounding box not computed correctly")

    # # now test with 3d cubes
    # data = np.ones((2,2,2))
    # pi = models.PixelIsland(dim=3)
    # pi.calc_bounding_box(data, offsets=[0,0,0])
    # if not np.all(pi.bounding_box == [[0,2],[0,2],[0,2]]):
    #     raise AssertionError("bounding box not computed correctly for 3d data")


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
