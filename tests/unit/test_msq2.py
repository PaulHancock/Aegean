#! /usr/bin/env python
"""
Test msq2.py
"""
import numpy as np
from AegeanTools.msq2 import MarchingSquares

__author__ = 'Paul Hancock'


def test_defaults():
    """Test that we can do a basic task"""
    # make a + shape from ones on a background of nan
    data = np.zeros((5, 5))*np.nan
    data[1:4, 2] = 1
    data[2, 1:4] = 1
    ms = MarchingSquares(data)
    if not (ms.xsize == ms.ysize == 5):
        raise AssertionError()
    if not (len(ms.perimeter) == 12):
        raise AssertionError()
    if not (ms.find_start_point() == (1, 2)):
        raise AssertionError()


def test_multi_islands():
    """Test with two islands"""
    # make a C and a + on a background of zeros
    data = np.zeros((7, 9), dtype=int)
    data[1, [1, 2, 3]] = 1
    data[2, [1, 3, 6]] = 1
    data[3, [1, 5, 6, 7]] = 1
    data[4, :] = data[2, :]
    data[5, :] = data[1, :]
    ms = MarchingSquares(data)
    if not (np.all(ms.data == data)):
        raise AssertionError()
    perims = ms.do_march_all()
    if not (len(perims) == 2):
        raise AssertionError()
    if not (np.all(ms.data == data)):
        raise AssertionError()


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
