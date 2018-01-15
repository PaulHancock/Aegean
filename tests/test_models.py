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
    assert isinstance(ss.ra, np.float64)
    # convert to string without failing
    a = "{0}".format(ss)
    assert a == ' 12.0000000   0.0000000  0.000000  0.000000  0.00  0.00    0.0 000000'
    assert ss.__repr__() == ss.__str__()
    assert np.all(ss.as_list()[:-1] == [0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0])
    isl = models.IslandSource()
    isl2 = models.IslandSource()
    assert isl < ss
    assert isl <= ss
    assert not(isl > ss)
    assert not(isl >= ss)
    assert isl == isl2
    assert not(isl != isl2)
    out = models.OutputSource()
    out.source = 1
    out2 = models.OutputSource()
    out2.source = 2
    out3 = models.OutputSource()
    out3.island = 1
    assert out < out2
    assert out3 > out2
    assert out <= out2
    assert out3 >= out
    assert out != out2
    assert out == out


def test_global_fitting_data():
    models.GlobalFittingData()


def test_island_fitting_data():
    models.IslandFittingData()


def test_classify_catalogue():
    ss = []
    isl = []
    out = []
    all = []
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
        all.extend([a, b, c, d])
    a, b, c = models.classify_catalog(all)
    assert np.all(b == isl)
    assert np.all(a == out)
    groups = list(models.island_itergen(a))
    assert len(groups) == 10


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()