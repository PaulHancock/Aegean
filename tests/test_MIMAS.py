#! /usr/bin/env python
"""
Test MIMAS.py
"""

from __future__ import print_function
from AegeanTools import MIMAS
import numpy as np
import os

__author__ = 'Paul Hancock'


def test_Dummy():
    """test the Dummy class"""
    a = MIMAS.Dummy()
    return a


def test_galactic2fk5():
    """test function galactic2fk5"""
    l, b = 12, 0
    ra, dec = MIMAS.galactic2fk5(l, b)
    return ra, dec


def test_mask_plane():
    """TODO"""
    return


def test_mask_file():
    """TODO"""
    return


def test_mask_table():
    """TODO"""
    return


def test_mask_catalog():
    """TODO"""
    return


def test_mim2reg():
    """TODO"""
    return


def test_mim2fits():
    """TODO"""
    return


def test_box2poly():
    """TODO"""
    return


def test_circle2circle():
    """TODO"""
    return


def test_poly2poly():
    """TODO"""
    return


def test_reg2mim():
    """TODO"""
    return


def test_combine_regions():
    """TODO"""
    return


def test_intersect_regions():
    """TODO"""
    return


def test_save_region():
    """TODO"""
    return


def test_save_as_image():
    """TODO"""
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()