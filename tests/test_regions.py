#! /usr/bin/env python
"""
Test regions.py
"""

from __future__ import print_function

__author__ = 'Paul Hancock'

from AegeanTools.regions import Region
import numpy as np
import os


def test_radec2sky():
    """Test function: Region.radec2sky"""
    ra, dec = (15, -45)
    sky = Region.radec2sky(ra, dec)
    if not (np.all(sky == [(ra, dec)])): raise AssertionError("radec2sky broken on non-list input")
    ra = [0, 10]
    dec = [-45, 45]
    sky = Region.radec2sky(ra, dec)
    answer = np.array([(ra[0], dec[0]), (ra[1], dec[1])])
    if not (np.all(sky == answer)): raise AssertionError('radec2sky broken on list input')


def test_sky2ang_symmetric():
    """Test that function Region.sky2ang is symmetric"""
    sky = np.radians(np.array([[15, -45]]))
    tp = Region.sky2ang(sky)
    tp = np.array([[tp[0][1], tp[0][0]]])
    sky2 = Region.sky2ang(tp)
    sky2 = np.array([[sky2[0][1], sky2[0][0]]])
    if not (np.all(abs(sky-sky2) < 1e-9)): raise AssertionError("sky2ang failed to be symmetric")


def test_sky2ang_corners():
    """Test that function Region.sky2ang works at 0/0 and the south pole"""
    corners = np.radians([[0, 0], [360, -90]])
    theta_phi = Region.sky2ang(corners)
    answers = np.array([[np.pi/2, 0], [np.pi, 2*np.pi]])
    if not (np.all(theta_phi - answers < 1e-9)): raise AssertionError('sky2ang corner cases failed')


def test_sky2vec_corners():
    """Test that function Region.sky2vec works at some tricky locations"""
    sky = np.radians([[0, 0], [90, 90], [45, -90]])
    answers = np.array([[1, 0, 0], [0, 0, 1], [0, 0, -1]])
    vec = Region.sky2vec(sky)
    if not (np.all(vec - answers < 1e-9)): raise AssertionError('sky2vec corner cases failed')


def test_vec2sky_corners():
    """Test that function Region.vec2sky works at some tricky locations"""
    vectors = np.array([[1, 0, 0], [0, 0, 1], [0, 0, -1]])
    skycoords = Region.vec2sky(vectors, degrees=True)
    answers = np.array([[0, 0], [0, 90], [0, -90]])
    if not (np.all(skycoords == answers)): raise AssertionError('vec2sky fails on corners')


def test_sky2vec2sky():
    """Test that function Region.vec2sky and Region.sky2vec are mutual inverses"""
    ra, dec = np.radians(np.array((0, -45)))
    sky = Region.radec2sky(ra, dec)
    vec = Region.sky2vec(sky)
    sky2 = Region.vec2sky(vec)
    if not (np.all(np.array(sky2) - np.array(sky) == 0)): raise AssertionError("sky2vec2sky failed")
    vec2 = Region.sky2vec(sky2)
    if not (np.all(np.array(vec) - np.array(vec2) == 0)): raise AssertionError('vec2sky2vec failed')


def test_add_circles_list_scalar():
    """Test that Region.add_circles works for vector inputs"""
    ra_list = np.radians([13.5, 13.5])
    dec_list = np.radians([-90, -90])
    radius_list = np.radians([0.1, 0.01])
    ra = ra_list[0]
    dec = dec_list[0]
    radius = radius_list[0]
    region1 = Region(maxdepth=11)
    region2 = Region(maxdepth=11)
    region1.add_circles(ra_list, dec_list, radius_list)
    region1._demote_all()
    region2.add_circles(ra, dec, radius)
    region2._demote_all()
    test = True
    for i in range(1, region1.maxdepth+1):
        if len(region1.pixeldict[i].difference(region2.pixeldict[i])) > 0:
            test = False
    if not (test): raise AssertionError('add_circles gives different results for lists and scalars')


def test_renorm_demote_symmetric():
    """Test that Region._renorm and Region._demote are mutual inverses"""
    ra = 13.5
    dec = -90
    radius = 0.1
    region = Region(maxdepth=11)
    region.add_circles(np.radians(ra), np.radians(dec), np.radians(radius))
    region._demote_all()
    start_dict = region.pixeldict.copy()
    region._renorm()
    region._demote_all()
    end_dict = region.pixeldict.copy()
    test = True
    for i in range(1, region.maxdepth+1):
        if len(end_dict[i].difference(start_dict[i])) > 0:
            test = False
    if not (test): raise AssertionError('renorm and demote are not symmetric')


def test_sky_within():
    """Test the Ragion.sky_within method"""
    ra = np.radians([13.5, 15])
    dec = np.radians([-45, -40])
    radius = np.radians([0.1, 0.1])
    region = Region(maxdepth=11)
    region.add_circles(ra, dec, radius)
    if not (np.all(region.sky_within(ra[0], dec[0]))): raise AssertionError("Failed on position at center of region")
    if not (np.all(region.sky_within(ra, dec))): raise AssertionError("Failed on list of positions")
    if np.any(region.sky_within(ra[0]+5*radius[0], dec[0])): raise AssertionError("Failed on position outside of region")
    try:
        region.sky_within(np.nan, dec[0])
    except ValueError as e:
        raise AssertionError("Failed with a nan position\n" + e.message)


def test_pickle():
    """ Test that the Region class can be pickled and loaded without loss """
    ra = 66.38908
    dec = -26.72466
    radius = 22
    region = Region(maxdepth=8)
    region.add_circles(np.radians(ra), np.radians(dec), np.radians(radius))
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    pickle.dump(region, open('out_temp.mim', 'wb'))
    region2 = pickle.load(open('out_temp.mim','rb'))
    if not (region.pixeldict == region2.pixeldict): raise AssertionError('pickle/unpickle does not give same region')
    os.remove('out_temp.mim')


def test_reg():
    """
    Test that .reg files can be written without crashing
    (Not a test that the .reg files are valid)
    """
    ra = np.radians([285])
    dec = np.radians([-66])
    radius = np.radians([3])
    region = Region(maxdepth=5)
    region.add_circles(ra, dec, radius)
    region.write_reg('test.reg')
    if not (os.path.exists('test.reg')): raise AssertionError()
    os.remove('test.reg')


def test_poly():
    """
    Test that polygon regions can be added and written to .reg files
    """
    ra = [5., 5., 7., 7.]
    dec = [-2., -2.5, -2.5, -2.]
    region = Region(maxdepth=6)
    positions = list(zip(np.radians(ra), np.radians(dec)))
    region.add_poly(positions)
    region.write_reg('test.reg')
    if not (os.path.exists('test.reg')): raise AssertionError()
    os.remove('test.reg')


def test_write_fits():
    """ Test that MOC files can be written in fits format """
    a = Region()
    a.add_circles(12, 0, 0.1)
    a.write_fits('test_MOC.fits')
    if not (os.path.exists('test_MOC.fits')): raise AssertionError()
    os.remove('test_MOC.fits')


def test_without():
    """
    Test the Region.without gives expected results"
    """
    a = Region(maxdepth=7)
    a.add_circles(0, np.radians(-90), np.radians(1))
    area = a.get_area()
    b = Region(maxdepth=7)
    b.add_circles(0, np.radians(-90), np.radians(0.5))
    a.without(b)
    if not (a.get_area() <= (area - b.get_area())): raise AssertionError("test_without FAILED")


def test_intersect():
    """
    Test the Region.intersect gives expected results"
    """
    a = Region(maxdepth=7)
    a.add_circles(0, np.radians(-90), np.radians(1))
    b = Region(maxdepth=7)
    b.add_circles(0, np.radians(-90), np.radians(0.5))
    a.intersect(b)
    if not (a.get_area() == b.get_area()):
        raise AssertionError("test_intersect FAILED")

    a = Region(maxdepth=8)
    a.add_circles(0, np.radians(75), np.radians(3))
    c = Region(maxdepth=8)
    c.add_circles(0, np.radians(90), np.radians(10))
    a.intersect(c)
    if not (a.get_area() == 0.):
        raise AssertionError("test_intersect FAILED")


def test_demote():
    """Test that we can demote a region"""
    a = Region(maxdepth=8)
    a.add_circles(0, np.radians(-90), np.radians(1))
    _ = a.pixeldict.copy()
    fpd = a.get_demoted()
    if not (fpd == a.pixeldict[8]): raise AssertionError()
    for i in range(1, 8):
        if not (len(a.pixeldict[i]) == 0): raise AssertionError()


def test_symmetric_difference():
    """
    Test the Region.symmetric_difference() gives expected results"
    """
    a = Region(maxdepth=7)
    a.add_circles(0, np.radians(-90), np.radians(1))
    area = a.get_area()
    b = Region(maxdepth=7)
    b.add_circles(0, np.radians(-90), np.radians(0.5))
    a.symmetric_difference(b)
    if not (a.get_area() == area - b.get_area()): raise AssertionError("test_symmetric_difference FAILED")


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
