#! /usr/bin/env python
"""
Test MIMAS.py
"""
from AegeanTools import MIMAS
import numpy as np
import os

__author__ = 'Paul Hancock'


def test_Dummy():
    """Test dummy class"""
    try:
        d = MIMAS.Dummy()
    except:
        raise AssertionError("Dummy.__init__ is broken")
    return


def test_galactic2fk5():
    """test function"""
    try:
        l, b = np.radians(13), np.radians(-41)
        ra, dec = MIMAS.galactic2fk5(l, b)
    except:
        raise AssertionError("galactic2fk5 is broken")
    return


def no_test_mask_plane():
    # TODO
    return


def no_test_mask_file():
    # TODO
    return


def no_test_mask_table():
    # TODO
    return


def no_test_mask_catalog():
    # TODO
    return


def no_test_mim2reg():
    # TODO
    return


def no_test_mim2fits():
    # TODO
    return


def test_mask2mim():
    maskfile = 'tests/test_files/mask.fits'
    mimfile = 'out.mim'
    # test with threshold of 1.0
    MIMAS.mask2mim(maskfile=maskfile,
                   mimfile=mimfile)
    if not os.path.exists(mimfile):
        raise AssertionError(
            "Failed to convert {0}->{1}".foramt(maskfile, mimfile))
    os.remove(mimfile)

    # test with threshold of 2.0
    MIMAS.mask2mim(maskfile=maskfile,
                   mimfile=mimfile,
                   threshold=2.0,
                   maxdepth=9)
    if not os.path.exists(mimfile):
        raise AssertionError(
            "Failed to convert {0}->{1}".foramt(maskfile, mimfile))
    os.remove(mimfile)

    return


def no_test_box2poly():
    # TODO
    return


def no_test_circle2circle():
    # TODO
    return


def no_test_poly2poly():
    # TODO
    return


def no_test_reg2mim():
    # TODO
    return


def no_test_combine_regions():
    # TODO
    return


def no_test_intersect_regions():
    # TODO
    return


def no_test_save_region():
    # TODO
    return


def no_test_save_as_image():
    # TODO
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
