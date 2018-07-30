#! /usr/bin/env python
"""
Test the AeRes module
"""

from __future__ import print_function

__author__ = 'Paul Hancock'

from AegeanTools import AeRes as ar
from AegeanTools import wcs_helpers

from astropy.io import fits
import numpy as np
import os


def test_load_sources():
    """Test load_sources"""
    filename = 'tests/test_files/1904_comp.fits'
    cat = ar.load_sources(filename)
    if cat is None:
        raise AssertionError("load_sources_failed")
    return


def test_make_model():
    """Test make_modell"""
    filename = 'tests/test_files/1904_comp.fits'
    sources = ar.load_sources(filename)
    hdulist = fits.open('tests/test_files/1904-66_SIN.fits')
    wcs_helper = wcs_helpers.WCSHelper.from_header(header=hdulist[0].header)
    # regular run
    model = ar.make_model(sources=sources, shape=hdulist[0].data.shape,
                          wcshelper=wcs_helper)
    if np.all(model == 0.):
        raise AssertionError("Model is empty")

    # model with *all* sources outside region
    # shape (100,2) means we only sometimes reject a source based on it's x-coord
    model = ar.make_model(sources=sources, shape=(100, 2),
                          wcshelper=wcs_helper)
    if not np.all(model == 0.):
        raise AssertionError("Model is *not* empty")

    # test mask with sigma
    model = ar.make_model(sources=sources, shape=hdulist[0].data.shape,
                          wcshelper=wcs_helper, mask=True)
    if np.all(model == 0.):
        raise AssertionError("Model is empty")
    if not np.any(np.isnan(model)):
        raise AssertionError("Model is not masked")

    # test mask with frac
    model = ar.make_model(sources=sources, shape=hdulist[0].data.shape,
                          wcshelper=wcs_helper, mask=True, frac=0.1)
    if np.all(model == 0.):
        raise AssertionError("Model is empty")
    if not np.any(np.isnan(model)):
        raise AssertionError("Model is not masked")


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
        raise AssertionError("Mask file not written")
    os.remove(masked)

    # with masking
    ar.make_residual(fitsfile=fitsfile, catalog=catalog,
                     rfile=residual, mfile=masked, mask=True)
    if not os.path.exists(masked):
        raise AssertionError("Mask file not written")
    os.remove(masked)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()