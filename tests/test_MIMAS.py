#! /usr/bin/env python
"""
Test MIMAS.py
"""
from AegeanTools import MIMAS
from AegeanTools.regions import Region
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


def test_mask_plane():
    # This is implicitly done in test_mask_file so just pass for now
    return


def test_mask_file():
    infile = 'tests/test_files/1904-66_SIN.fits'
    outfile = 'test_masked.fits'
    rfile = 'circle.mim'
    region = Region(maxdepth=8)
    region.add_circles(np.radians(285), np.radians(-65), 1.8)

    # check for failure in relevant conditions
    try:
        MIMAS.mask_file('nofile', 'nofile', 'nofile')
    except AssertionError as e:
        if not 'fits file' in e.args[0]:
            raise AssertionError('Failed to catch file not found err (image)')
    
    try:
        MIMAS.mask_file('nofile', infile,'nofile')
    except AssertionError as e:
        if not 'region file' in e.args[0]:
            raise AssertionError('Faile to catch file not found err (region)')
    
    # make a region file and test that a masked outfile can be made
    region.save(rfile)

    MIMAS.mask_file(rfile, infile, outfile)
    if not os.path.exists(outfile):
        os.remove(rfile)
        raise AssertionError("Failed to create masked file")
    os.remove(outfile)

    MIMAS.mask_file(rfile, infile, outfile, negate=True)
    if not os.path.exists(outfile):
        os.remove(rfile)
        raise AssertionError("Failed to create masked file")

    # cleanup
    os.remove(rfile)
    os.remove(outfile)
    return


def no_test_mask_table():
    # TODO
    return


def no_test_mask_catalog():
    # TODO
    return


def test_mim2reg():
    rfile = 'circle.mim'
    regfile = 'circle.reg'
    region = Region(maxdepth=8)
    region.add_circles(np.radians(285), np.radians(-65), 1.8)
    region.save(rfile)
    MIMAS.mim2reg(rfile, regfile)
    if not os.path.exists(regfile):
        os.remove(rfile)
        raise AssertionError("Failed to convert mim2reg")
    os.remove(rfile)
    os.remove(regfile)
    return


def test_mim2fits():
    rfile = 'circle.mim'
    fitsfile = 'circle.fits'
    region = Region(maxdepth=8)
    region.add_circles(np.radians(285), np.radians(-65), 1.8)
    region.save(rfile)
    MIMAS.mim2fits(rfile, fitsfile)
    if not os.path.exists(fitsfile):
        os.remove(rfile)
        raise AssertionError("Failed to convert mim2fits")
    os.remove(rfile)
    os.remove(fitsfile)
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
