#! /usr/bin/env python
"""
Test MIMAS.py
"""

from __future__ import print_function
from AegeanTools import MIMAS
from AegeanTools.regions import Region
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
    """test the intersection of multiple regions"""
    cap = Region()
    cap.add_circles(0, np.radians(-90), np.radians(10))
    cfile = 'cap.mim'
    MIMAS.save_region(cap, cfile)

    # MIMAS should complain about just one file to load
    try:
        MIMAS.intersect_regions([cfile])
    except Exception as e:
        pass
    else:
        raise AssertionError()

    # intersect a region with itself should produce same area/pixels
    cap2 = MIMAS.intersect_regions([cfile, cfile])
    if cap2.get_area() != cap.get_area():
        raise AssertionError("intersect broken on reflexive test")
    if not np.all(cap2.demoted == cap.demoted):
        raise AssertionError("intersect broken on reflexive test")

    # a new region near the equator with no overlap
    cap2 = Region()
    cap2.add_circles(0, 0, np.radians(3))
    c2file = 'cap2.mim'
    MIMAS.save_region(cap2, c2file)

    # the intersection should have no area
    i = MIMAS.intersect_regions([cfile, c2file])
    if not (i.get_area() == 0.):
        raise AssertionError("intersection doesn't give null result")

    os.remove(cfile)
    os.remove(c2file)
    return


def test_save_region():
    """test save_region"""
    cap = Region()
    cap.add_circles(0, np.radians(-90), np.radians(10))
    cfile = 'cap.mim'
    MIMAS.save_region(cap, cfile)
    if not os.path.exists(cfile):
        raise AssertionError("Failed to write file")
    os.remove(cfile)
    return


def test_save_as_image():
    """test save_as_image"""
    cap = Region()
    cap.add_circles(0, np.radians(30), np.radians(10))
    cfile = 'circle.png'
    MIMAS.save_as_image(cap, cfile)
    if not os.path.exists(cfile):
        raise AssertionError("Failed to write file")
    os.remove(cfile)
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()