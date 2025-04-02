from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pytest
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from radio_beam import Beam

from AegeanTools.BANE_fft import bane_fft, gaussian_kernel, pad_reflect, robust_bane


rng = np.random.default_rng(12345)
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")


class RandomArrays(NamedTuple):
    """
    Class to hold random arrays and their parameters.
    """

    arrays: list[NDArray[np.float32]]
    locs: NDArray[np.float64]
    scales: NDArray[np.float64]


@pytest.fixture
def ranmdom_arrays() -> RandomArrays:
    n_arrays = 10
    locs = rng.uniform(0, 100, size=n_arrays)
    scales = rng.uniform(0, 10, size=n_arrays)
    arrays = [
        rng.normal(loc=locs[i], scale=scales[i], size=(1024, 1024)).astype(np.float32)
        for i in range(n_arrays)
    ]
    return RandomArrays(arrays, locs, scales)


def test_pad_reflect(ranmdom_arrays: RandomArrays):
    """
    Test the pad function.
    """

    for array in ranmdom_arrays.arrays:
        array_pad = pad_reflect(
            array=array.astype(np.float32),
            pad_width=(10, 10),
        )
        array_pad_np = np.pad(
            array=array.astype(np.float32),
            pad_width=(10, 10),
            mode="reflect",
        )
        expected_shape = (array.shape[0] + 20, array.shape[1] + 20)
        assert np.array_equal(array_pad, array_pad_np), (
            "Pad function with reflect mode failed"
        )
        assert array_pad.shape == expected_shape, (
            f"Bad shape {array_pad.shape} != {expected_shape}"
        )


def test_bane_fft(ranmdom_arrays: RandomArrays):
    gaussian_kernel_arr = gaussian_kernel(10)
    for array, loc, scale in zip(
        ranmdom_arrays.arrays, ranmdom_arrays.locs, ranmdom_arrays.scales
    ):
        bkg, rms = bane_fft(
            image=array,
            kernel=gaussian_kernel_arr,
        )

        assert bkg.shape == array.shape, f"Bad shape {bkg.shape} != {array.shape}"
        assert rms.shape == array.shape, f"Bad shape {rms.shape} != {array.shape}"
        assert np.isclose(np.nanmean(bkg), loc, rtol=0.1), (
            "Mean of background is not close to the expected value"
        )
        assert np.isclose(np.nanmean(rms), scale, rtol=0.1), (
            "Mean of RMS is not close to the expected value"
        )


def test_robust_bane(ranmdom_arrays: RandomArrays):
    """
    Test the robust bane function.
    """
    for array, loc, scale in zip(
        ranmdom_arrays.arrays, ranmdom_arrays.locs, ranmdom_arrays.scales
    ):
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = array.shape[1]
        header["NAXIS2"] = array.shape[0]
        header["CRPIX1"] = array.shape[1] / 2
        header["CRPIX2"] = array.shape[0] / 2
        header["CRVAL1"] = 0
        header["CRVAL2"] = 0
        header["CDELT1"] = 2.5 / 3600
        header["CDELT2"] = -2.5 / 3600
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        wcs = WCS(header)
        header = wcs.to_header()
        beam = Beam(20 * u.arcsec, 20 * u.arcsec, 0.0 * u.deg)
        header = beam.attach_to_header(header)
        bkg, rms = robust_bane(
            image=array,
            header=header,
            step_size=0,
            clip_sigma=np.inf,
        )

        assert bkg.shape == array.shape, f"Bad shape {bkg.shape} != {array.shape}"
        assert rms.shape == array.shape, f"Bad shape {rms.shape} != {array.shape}"
        assert np.isclose(np.nanmean(bkg), loc, rtol=0.1), (
            "Mean of background is not close to the expected value"
        )
        assert np.isclose(np.nanmean(rms), scale, rtol=0.1), (
            "Mean of RMS is not close to the expected value"
        )
