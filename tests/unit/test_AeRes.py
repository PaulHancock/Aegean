#! /usr/bin/env python
"""
Test the AeRes module
"""
import os

import numpy as np
from AegeanTools import AeRes as ar
from AegeanTools import catalogs, wcs_helpers
from astropy.io import fits

__author__ = "Paul Hancock"


def test_load_sources():
    """Test load_sources"""
    filename = "tests/test_files/1904_comp.fits"
    cat = ar.load_sources(filename)
    if cat is None:
        raise AssertionError("load_sources failed")
    return


def test_load_soruces_with_alpha():
    """Test load_sources with alpha column"""
    filename = "tests/test_files/synthetic_with_alpha_comp.fits"
    cat = ar.load_sources(filename)
    if cat is None:
        raise AssertionError("load_sources failed with alpha column")
    return


def test_load_soruces_renamed_columns():
    """Test load_sources with renamed columns"""
    filename = "tests/test_files/1904_comp_renamed_cols.fits"
    colnames = {
        "ra_col": "RAJ2000",
        "dec_col": "DEJ2000",
        "peak_col": "S",
        "a_col": "bmaj",
        "b_col": "bmin",
        "pa_col": "bpa",
    }
    cat = ar.load_sources(filename, **colnames)
    if cat is None:
        raise AssertionError("load_sources failed with renamed columns")
    return


def test_load_sources_missing_columns():
    filename = "tests/test_files/1904_comp.fits"
    table = catalogs.load_table(filename)
    table.rename_column("ra", "RAJ2000")
    table.write("dlme.fits")
    cat = ar.load_sources("dlme.fits")
    if os.path.exists("dlme.fits"):
        os.remove("dlme.fits")

    if cat is not None:
        raise AssertionError("Missing columns should be caught, but weren't")
    return


def test_make_model():
    """Test make_model"""
    filename = "tests/test_files/1904_comp.fits"
    sources = ar.load_sources(filename)
    hdulist = fits.open("tests/test_files/1904-66_SIN.fits")
    wcs_helper = wcs_helpers.WCSHelper.from_header(header=hdulist[0].header)
    # regular run
    model = ar.make_model(
        sources=sources, shape=(1, *hdulist[0].data.shape), wcshelper=wcs_helper
    )
    if np.all(model == 0.0):
        raise AssertionError("Model is empty")

    # model with *all* sources outside region
    # shape (100,2) means we only sometimes reject a source based on it's x-coord
    model = ar.make_model(sources=sources, shape=(1, 100, 2), wcshelper=wcs_helper)
    if not np.all(model == 0.0):
        raise AssertionError("Model is *not* empty")


def test_make_model_with_alpha():
    """Test make_model with alpha column"""
    filename = "tests/test_files/synthetic_with_alpha_comp.fits"
    sources = ar.load_sources(filename)
    hdulist = fits.open("tests/test_files/synthetic_with_alpha.fits")
    wcs_helper = wcs_helpers.WCSHelper.from_header(header=hdulist[0].header)
    # regular run
    model = ar.make_model(
        sources=sources, shape=hdulist[0].data.shape, wcshelper=wcs_helper
    )
    if np.all(model == 0.0):
        raise AssertionError("Model is empty")

    # model with *all* sources outside region
    # shape (100,2) means we only sometimes reject a source based on it's x-coord
    model = ar.make_model(sources=sources, shape=(1, 100, 2), wcshelper=wcs_helper)
    if not np.all(model == 0.0):
        raise AssertionError("Model is *not* empty")


def test_make_masked_model():
    """Test make_model when a mask is being used"""
    filename = "tests/test_files/1904_comp.fits"
    sources = ar.load_sources(filename)
    hdulist = fits.open("tests/test_files/1904-66_SIN.fits")
    wcs_helper = wcs_helpers.WCSHelper.from_header(header=hdulist[0].header)

    # test mask with sigma
    model = ar.make_model(
        sources=sources,
        shape=(1, *hdulist[0].data.shape),
        wcshelper=wcs_helper,
        mask=True,
    )

    finite = np.where(np.isfinite(model))
    if np.all(model == 0.0):
        raise AssertionError("Model is empty")
    if not np.any(np.isnan(model)):
        raise AssertionError("Model is not masked")
    if not np.all(model[finite] == 0.0):
        raise AssertionError("Model has values that are not zero or nan")

    # test mask with frac
    model = ar.make_model(
        sources=sources,
        shape=(1, *hdulist[0].data.shape),
        wcshelper=wcs_helper,
        mask=True,
        frac=0.1,
    )

    finite = np.where(np.isfinite(model))
    if np.all(model == 0.0):
        raise AssertionError("Model is empty")
    if not np.any(np.isnan(model)):
        raise AssertionError("Model is not masked")
    if not np.all(model[finite] == 0.0):
        raise AssertionError("Model has values that are not zero or nan")


def test_make_residual():
    """Test make_residual"""
    fitsfile = "tests/test_files/1904-66_SIN.fits"
    catalog = "tests/test_files/1904_comp.fits"
    residual = "tests/temp/residual.fits"
    masked = "tests/temp/masked.fits"
    # default operation
    ar.make_residual(
        fitsfile=fitsfile, catalog=catalog, rfile=residual, mfile=masked, mask=False
    )
    if not os.path.exists(masked):
        raise AssertionError("Mask file not written")
    os.remove(masked)

    # with masking
    ar.make_residual(
        fitsfile=fitsfile, catalog=catalog, rfile=residual, mfile=masked, mask=True
    )
    if not os.path.exists(masked):
        raise AssertionError("Mask file not written")
    os.remove(masked)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
