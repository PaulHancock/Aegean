#! python

__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import models
import numpy as np


def test_simple_source():
    # make a new source without failing
    ss = models.SimpleSource()
    ss.ra = np.float32(12)
    ss._sanitise()
    if not (isinstance(ss.ra, np.float64)): raise AssertionError()
    # convert to string without failing
    a = "{0}".format(ss)
    if not (a == ' 12.0000000   0.0000000  0.000000  0.000000  0.00  0.00    0.0 000000'): raise AssertionError()
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
    out = models.OutputSource()
    out.source = 1
    out2 = models.OutputSource()
    out2.source = 2
    out3 = models.OutputSource()
    out3.island = 1
    if not (out < out2): raise AssertionError()
    if not (out3 > out2): raise AssertionError()
    if not (out <= out2): raise AssertionError()
    if not (out3 >= out): raise AssertionError()
    if not (out != out2): raise AssertionError()
    if not (out == out): raise AssertionError()


def test_global_fitting_data():
    models.GlobalFittingData()


def test_island_fitting_data():
    models.IslandFittingData()


def test_classify_catalogue():
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
        c = models.OutputSource()
        c.island = i
        d = models.OutputSource()
        d.island = i
        d.source = 1
        out.extend([c, d])
        mixed.extend([a, b, c, d])
    a, b, c = models.classify_catalog(mixed)
    if not (np.all(b == isl)): raise AssertionError()
    if not (np.all(a == out)): raise AssertionError()
    groups = list(models.island_itergen(a))
    if not (len(groups) == 10): raise AssertionError()


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()