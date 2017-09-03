#! python
from __future__ import print_function

from AegeanTools.msq2 import MarchingSquares
import numpy as np

__author__ = 'Paul Hancock'
__date__ = ''


def test_defaults():
    # make a + shape from ones on a background of nan
    data = np.zeros((5, 5))*np.nan
    data[1:4, 2] = 1
    data[2, 1:4] = 1
    ms = MarchingSquares(data)
    assert ms.xsize == ms.ysize == 5
    assert len(ms.perimeter) == 12
    assert ms.find_start_point() == (1, 2)


def test_multi_islands():
    # make a C and a cap on a background of zeros
    data = np.zeros((7, 9), dtype=np.int)
    data[1, [1, 2, 3, 5, 6, 7]] = 1
    data[2, [1, 3, 5, 7]] = 1
    data[3, 1] = 1
    data[4, :] = data[2, :]
    data[5, :] = data[1, :]
    ms = MarchingSquares(data)
    assert np.all(ms.data == data)
    perims = ms.do_march_all()
    assert len(perims) == 3
    assert np.all(ms.data == data)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")