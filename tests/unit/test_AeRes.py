#! /usr/bin/env python
"""
Test the AeRes module
"""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from treasure_island import AeRes as ar
from treasure_island import catalogs, wcs_helpers

__author__ = 'Paul Hancock'


def test_load_sources():
    """Test load_sources"""
    filename = 'tests/test_files/1904_comp.fits'
    cat = ar.load_sources(filename)
    if cat is None:
        msg = "load_sources failed"
        raise AssertionError(msg)


def test_load_soruces_renamed_columns():
    """Test load_sources with renamed columns"""
    filename = 'tests/test_files/1904_comp_renamed_cols.fits'
    colnames = {'ra_col': 'RAJ2000',
                'dec_col': 'DEJ2000',
                'peak_col': 'S',
                'a_col': 'bmaj',
                'b_col': 'bmin',
                'pa_col': 'bpa'}
    cat = ar.load_sources(filename, **colnames)
    if cat is None:
        msg = "load_sources failed with renamed columns"
        raise AssertionError(msg)


def test_load_sources_missing_columns():
    filename = 'tests/test_files/1904_comp.fits'
    table = catalogs.load_table(filename)
    table.rename_column('ra', 'RAJ2000')
    table.write('dlme.fits')
    cat = ar.load_sources('dlme.fits')
    if os.path.exists('dlme.fits'):
        os.remove('dlme.fits')

    if cat is not None:
        msg = "Missing columns should be caught, but weren't"
        raise AssertionError(msg)


def test_make_model():
    """Test make_model"""
    filename = 'tests/test_files/1904_comp.fits'
    sources = ar.load_sources(filename)
    hdulist = fits.open('tests/test_files/1904-66_SIN.fits')
    wcs_helper = wcs_helpers.WCSHelper.from_header(header=hdulist[0].header)
    # regular run
    model = ar.make_model(sources=sources, shape=hdulist[0].data.shape,
                          wcshelper=wcs_helper)
    if np.all(model == 0.):
        msg = "Model is empty"
        raise AssertionError(msg)

    # model with *all* sources outside region
    # shape (100,2) means we only sometimes reject a source based on it's x-coord
    model = ar.make_model(sources=sources, shape=(100, 2),
                          wcshelper=wcs_helper)
    if not np.all(model == 0.):
        msg = "Model is *not* empty"
        raise AssertionError(msg)


def test_make_masked_model():
    """Test make_model when a mask is being used"""
    filename = 'tests/test_files/1904_comp.fits'
    sources = ar.load_sources(filename)
    hdulist = fits.open('tests/test_files/1904-66_SIN.fits')
    wcs_helper = wcs_helpers.WCSHelper.from_header(header=hdulist[0].header)

    # test mask with sigma
    model = ar.make_model(sources=sources, shape=hdulist[0].data.shape,
                          wcshelper=wcs_helper, mask=True)

    finite = np.where(np.isfinite(model))
    if np.all(model == 0.):
        msg = "Model is empty"
        raise AssertionError(msg)
    if not np.any(np.isnan(model)):
        msg = "Model is not masked"
        raise AssertionError(msg)
    if not np.all(model[finite] == 0.):
        msg = "Model has values that are not zero or nan"
        raise AssertionError(msg)

    # test mask with frac
    model = ar.make_model(sources=sources, shape=hdulist[0].data.shape,
                          wcshelper=wcs_helper, mask=True, frac=0.1)

    finite = np.where(np.isfinite(model))
    if np.all(model == 0.):
        msg = "Model is empty"
        raise AssertionError(msg)
    if not np.any(np.isnan(model)):
        msg = "Model is not masked"
        raise AssertionError(msg)
    if not np.all(model[finite] == 0.):
        msg = "Model has values that are not zero or nan"
        raise AssertionError(msg)


def test_make_residual():
    """Test make_residual"""
    fitsfile = 'tests/test_files/1904-66_SIN.fits'
    catalog = 'tests/test_files/1904_comp.fits'
    residual = 'tests/temp/residual.fits'
    masked = 'tests/temp/masked.fits'
    # default operation
    ar.make_residual(fitsfile=fitsfile, catalog=catalog,
                     rfile=residual, mfile=masked, mask=False)
    if not os.path.exists(masked):
        msg = "Mask file not written"
        raise AssertionError(msg)
    os.remove(masked)

    # with masking
    ar.make_residual(fitsfile=fitsfile, catalog=catalog,
                     rfile=residual, mfile=masked, mask=True)
    if not os.path.exists(masked):
        msg = "Mask file not written"
        raise AssertionError(msg)
    os.remove(masked)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
