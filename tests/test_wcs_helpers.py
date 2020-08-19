#! /usr/bin/envy python
"""
Test wcs_helpers.py
"""
from __future__ import print_function

from AegeanTools.wcs_helpers import WCSHelper, Beam
from astropy.io import fits
import numpy as np
from numpy.testing import assert_almost_equal

__author__ = 'Paul Hancock'


def verify_beam(beam):
    """fail if the given beam is not valid"""
    if beam is None: raise AssertionError()
    if not (beam.a > 0): raise AssertionError()
    if not (beam.b > 0): raise AssertionError()
    if beam.pa is None: raise AssertionError()


def test_from_header():
    """Test that we can make a beam from a fitsheader"""
    fname = 'tests/test_files/1904-66_SIN.fits'
    header = fits.getheader(fname)
    helper = WCSHelper.from_header(header)
    if helper.beam is None: raise AssertionError()
    del header['BMAJ'], header['BMIN'], header['BPA']
    # Raise an error when the beam information can't be determined
    try:
        _ = WCSHelper.from_header(header)
    except AssertionError as e:
        pass
    else:
        raise AssertionError("Header with no beam information should thrown an exception.")
    return


def test_from_file():
    """Test that we can load from a file"""
    fname = 'tests/test_files/1904-66_SIN.fits'
    helper = WCSHelper.from_file(fname)
    if helper.beam is None: raise AssertionError()


def test_get_pixbeam():
    """Test get_pixbeam"""
    fname = 'tests/test_files/1904-66_SIN.fits'
    helper = WCSHelper.from_file(fname)

    beam = Beam(*helper.get_psf_pix2pix(0, 0))
    verify_beam(beam)

    beam = helper.get_skybeam(285, -66)
    verify_beam(beam)

    beam = helper.get_skybeam(285, -66)
    verify_beam(beam)

    area = helper.get_beamarea_pix(285, -66)
    if not (area > 0): raise AssertionError()
    area = helper.get_beamarea_deg2(285, -66)
    if not (area >0): raise AssertionError()


def test_psf_files():
    """Test that using external psf files works properly"""

    fname = 'tests/test_files/1904-66_SIN.fits'
    psfname = 'tests/test_files/1904-66_SIN_psf.fits'
    helper = WCSHelper.from_file(fname, psf_file=psfname)

    # The psf image has only 19x19 pix and this ra/dec is
    a, b, pa = helper.get_psf_sky2sky(290., -63.)
    if not np.all(np.isfinite((a, b, pa))):
        raise AssertionError("Beam contains nans")


def test_sky_sep():
    """Test sky separation"""
    fname = 'tests/test_files/1904-66_SIN.fits'
    helper = WCSHelper.from_file(fname)
    dist = helper.sky_sep((0, 0), (1, 1))
    if not (dist > 0): raise AssertionError()


def test_vector_round_trip():
    """
    Converting a vector from pixel to sky coords and back again should give the
    original vector (within some tolerance).
    """
    fname = 'tests/test_files/1904-66_SIN.fits'
    helper = WCSHelper.from_file(fname)
    initial = [1, 45]  # r,theta = 1,45 (degrees)
    ref = helper.refpix
    ra, dec, dist, ang = helper.pix2sky_vec(ref, *initial)
    _, _ , r, theta = helper.sky2pix_vec((ra, dec), dist, ang)
    if not ((abs(r - initial[0]) < 1e-9) and (abs(theta - initial[1]) < 1e-9)): raise AssertionError()


def test_ellipse_round_trip():
    """
    Converting an ellipse from pixel to sky coords and back again should give the
    original ellipse (within some tolerance).
    """
    fname = 'tests/test_files/1904-66_SIN.fits'
    helper = WCSHelper.from_file(fname)
    a = 2 * helper.beam.a
    b = helper.beam.b
    pa = helper.beam.pa + 45
    ralist = list(range(-180, 180, 10))
    # SIN projection isn't valid for all decs
    declist = list(range(-85, -10, 10))
    ras, decs = np.meshgrid(ralist, declist)
    for _, (ra, dec) in enumerate(zip(ras.ravel(), decs.ravel())):
        if ra < 0:
            ra += 360
        x, y, sx, sy, theta = helper.sky2pix_ellipse((ra, dec), a, b, pa)
        ra_f, dec_f, major, minor, pa_f = helper.pix2sky_ellipse((x, y), sx, sy, theta)
        assert_almost_equal(ra, ra_f)
        assert_almost_equal(dec, dec_f)
        if not (abs(a-major)/a < 0.05): raise AssertionError()
        if not (abs(b-minor)/b < 0.05): raise AssertionError()
        if not (abs(pa-pa_f) < 1): raise AssertionError()


def test_psf_funcs():
    """
    For a SIN projected image, test that the beam/psf calculations are consistent at different points on the sky
    """
    fname = 'tests/test_files/1904-66_SIN.fits'
    helper = WCSHelper.from_file(fname)
    header = helper.wcs.to_header()

    refpix = header['CRPIX2'], header['CRPIX1']  # note x/y swap
    refcoord = header['CRVAL1'], header['CRVAL2']
    beam_ref_coord = helper.get_skybeam(*refcoord)
    beam_ref_pix = helper.get_skybeam(*helper.pix2sky(refpix))

    msg = 'sky beams at ref coord/pix disagree'
    if not abs(beam_ref_coord.a - beam_ref_pix.a) < 1e-5:
        raise AssertionError(msg)
    if not abs(beam_ref_coord.b - beam_ref_pix.b) < 1e-5:
        raise AssertionError(msg)
    if not abs(beam_ref_coord.pa - beam_ref_pix.pa) < 1e-5:
        raise AssertionError(msg)

    psf_zero = helper.get_psf_sky2pix(0,0)
    psf_refcoord = helper.get_psf_sky2pix(*refcoord)

    diff = np.subtract(psf_zero, psf_refcoord)
    if not np.all(diff < 1e-5):
        raise AssertionError("psf varies in pixel coordinates (it should not)")

    diff = np.subtract(psf_refcoord, (helper._psf_a, helper._psf_b, helper._psf_theta))
    if not np.all(diff < 1e-5):
        raise AssertionError("psf varies in pixel coordinates (it should not)")
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
