#! /usr/bin/env python
"""
Test MIMAS.py
"""
from AegeanTools import MIMAS
from AegeanTools.regions import Region
from astropy.table import Table
import numpy as np
import os

__author__ = "Paul Hancock"


def test_Dummy():
    """Test dummy class"""
    try:
        MIMAS.Dummy()
    except:
        raise AssertionError("Dummy.__init__ is broken")
    return


def test_galactic2fk5():
    """test function"""
    try:
        l, b = np.radians(13), np.radians(-41)
        MIMAS.galactic2fk5(l, b)
    except:
        raise AssertionError("galactic2fk5 is broken")
    return


def test_mask_plane():
    # This is implicitly done in test_mask_file so just pass for now
    return


def test_mask_file():
    infile = "tests/test_files/1904-66_SIN.fits"
    outfile = "test_masked.fits"
    rfile = "circle.mim"
    region = Region(maxdepth=8)
    region.add_circles(np.radians(285), np.radians(-65), 1.8)

    # check for failure in relevant conditions
    try:
        MIMAS.mask_file("nofile", "nofile", "nofile")
    except AssertionError as e:
        if "fits file" not in e.args[0]:
            raise AssertionError("Failed to catch file not found err (image)")

    try:
        MIMAS.mask_file("nofile", infile, "nofile")
    except AssertionError as e:
        if "region file" not in e.args[0]:
            raise AssertionError("Faile to catch file not found err (region)")

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


def test_mask_table():
    region = Region(maxdepth=8)
    region.add_circles(np.radians(285), np.radians(0), np.radians(2.1))
    ra = np.linspace(280, 290, 11)
    dec = np.zeros(11)
    tab = Table(data=[ra, dec], names=("ra", "dec"))
    masked = MIMAS.mask_table(region, tab)
    if len(masked) != 6:
        print(len(masked))
        raise AssertionError("failed to mask table correctly")

    masked = MIMAS.mask_table(region, tab, negate=True)
    if len(masked) != 5:
        raise AssertionError("failed to mask table correctly")
    return


def test_mask_catalog():
    infile = "tests/test_files/1904_comp.fits"
    regionfile = "tests/test_files/1904-66_SIN.mim"
    outfile = "dlme.fits"

    MIMAS.mask_catalog(regionfile, infile, outfile)
    if not os.path.exists(outfile):
        raise AssertionError("failed to mask catalogue")
    os.remove(outfile)

    MIMAS.mask_catalog(regionfile, infile, outfile, negate=True)
    if not os.path.exists(outfile):
        raise AssertionError("failed to mask catalogue")
    os.remove(outfile)

    return


def test_mim2reg():
    rfile = "circle.mim"
    regfile = "circle.reg"
    region = Region(maxdepth=4)
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
    rfile = "circle.mim"
    fitsfile = "circle.fits"
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
    maskfile = "tests/test_files/mask.fits"
    mimfile = "out.mim"
    # test with threshold of 1.0
    MIMAS.mask2mim(maskfile=maskfile, mimfile=mimfile)
    if not os.path.exists(mimfile):
        raise AssertionError("Failed to convert {0}->{1}".foramt(maskfile, mimfile))
    os.remove(mimfile)

    # test with threshold of 2.0
    MIMAS.mask2mim(maskfile=maskfile, mimfile=mimfile, threshold=2.0, maxdepth=9)
    if not os.path.exists(mimfile):
        raise AssertionError("Failed to convert {0}->{1}".foramt(maskfile, mimfile))
    os.remove(mimfile)

    return


def test_box2poly():
    box = 'box(290.3305929,-61.97230589,12720.000",10080.000",75.15517)'
    b = MIMAS.box2poly(box)
    if len(b) != 8:
        raise AssertionError("box2poly failed")
    return


def test_circle2circle():
    circle = 'circle(19:56:03.7988,-69:00:43.304,5755.555")'
    c = MIMAS.circle2circle(circle)
    if len(c) != 3:
        raise AssertionError("circle2circle failed")

    circle = 'circle(203.7988,-47,360")'
    c = MIMAS.circle2circle(circle)
    if len(c) != 3:
        raise AssertionError("circle2circle failed")
    return


def test_poly2poly():
    poly = "polygon(,,293.09,-71:13:55.511,19:28:20.4283,-69:05:34.569,19:20:41.1627,-69:00:08.332,19:10:57.4154,-70:03:28.840,18:43:29.5694,-67:46:43.665,19:03:21.1495,-72:00:50.438)"
    r = MIMAS.poly2poly(poly)
    if len(r) != 12:
        raise AssertionError("poly2poly failed")
    return


def test_reg2mim():
    reg = "tests/test_files/ds9.reg"
    outfile = "dlme.mim"
    MIMAS.reg2mim(reg, outfile, maxdepth=4)
    if not os.path.exists(outfile):
        raise AssertionError("reg2mim failed")
    os.remove(outfile)
    return


def test_combine_regions():
    file1 = "dlme1.mim"
    file2 = "dlme2.mim"
    r1 = Region(maxdepth=3)
    r1.add_circles(np.radians(285), np.radians(-65), 1.8)
    r1.save(file1)
    r1.save(file2)

    class Ob:
        pass

    c = Ob()
    c.maxdepth = 3
    c.add_region = [[file1]]
    c.rem_region = [[file2]]
    c.include_circles = [[12, -13, 1]]
    c.exclude_circles = [[10, 12, 3]]
    c.include_polygons = [[0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0]]
    c.exclude_polygons = [[0, 0, 0, 1, 1, 1, 1, 0]]

    c.galactic = False
    MIMAS.combine_regions(c)

    c.galactic = True
    MIMAS.combine_regions(c)

    os.remove(file1)
    os.remove(file2)
    return


def test_intersect_regions():
    file1 = "dlme1.mim"
    file2 = "dlme2.mim"

    try:
        MIMAS.intersect_regions([None])
    except Exception as e:
        if "Require" not in e.args[0]:
            raise AssertionError("intersect_regions failed")

    r1 = Region(maxdepth=3)
    r1.add_circles(np.radians(285), np.radians(-65), 1.8)
    r1.save(file1)
    r1.save(file2)

    a = MIMAS.intersect_regions([file1, file2])
    os.remove(file1)
    os.remove(file2)
    if not a:
        raise AssertionError("intersect_regions failed")

    return


def test_save_region():
    rfile = "circle.mim"
    region = Region(maxdepth=3)
    region.add_circles(np.radians(285), np.radians(-65), 1.8)
    MIMAS.save_region(region, rfile)
    if not os.path.exists(rfile):
        raise AssertionError("save_region failed")
    os.remove(rfile)
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
