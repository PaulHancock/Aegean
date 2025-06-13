from __future__ import annotations

import logging
from typing import NamedTuple

import astropy.units as u
import numpy as np
from numpy import fft
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray
from radio_beam import Beam

from AegeanTools.BANE_fft import (
    bane_fft,
    gaussian_kernel,
    pad_reflect,
    robust_bane,
    _ft_kernel,
)

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
        rng.normal(loc=locs[i], scale=scales[i], size=(512, 512)).astype(np.float32)
        for i in range(n_arrays)
    ]
    return RandomArrays(arrays, locs, scales)


@pytest.fixture
def ranmdom_arrays_odd() -> RandomArrays:
    n_arrays = 10
    locs = rng.uniform(0, 100, size=n_arrays)
    scales = rng.uniform(0, 10, size=n_arrays)
    arrays = [
        rng.normal(loc=locs[i], scale=scales[i], size=(1346, 2691)).astype(np.float32)
        for i in range(n_arrays)
    ]
    return RandomArrays(arrays, locs, scales)


def test_pad_reflect(ranmdom_arrays: RandomArrays):
    """
    Test the pad function.
    """

    pad_size = 10

    for array in ranmdom_arrays.arrays:
        array_pad = pad_reflect(
            array=array.astype(np.float32),
            pad_width=(pad_size, pad_size),
        )
        array_pad_np = np.pad(
            array=array.astype(np.float32),
            pad_width=(pad_size, pad_size),
            mode="reflect",
        )
        expected_shape = (array.shape[0] + pad_size * 2, array.shape[1] + pad_size * 2)
        assert np.array_equal(array_pad, array_pad_np), (
            "Pad function with reflect mode failed"
        )
        assert array_pad.shape == expected_shape, (
            f"Bad shape {array_pad.shape} != {expected_shape}"
        )


def test_pad_reflect_odd(ranmdom_arrays_odd: RandomArrays):
    """
    Test the pad function on odd-sized arrays.
    """

    pad_x, pad_y = (103, 103)
    for image in ranmdom_arrays_odd.arrays:
        image_padded = pad_reflect(
            array=image,
            pad_width=(pad_x, pad_y),
        )
        image_unpadded = image_padded[
            pad_x:-pad_x,
            pad_y:-pad_y,
        ]
        assert np.array_equal(image, image_unpadded), (
            "Pad function with reflect mode failed on odd-sized arrays"
        )
        assert image.shape == image_unpadded.shape


def test_fft_odd(ranmdom_arrays_odd: RandomArrays):
    """
    Test the FFT function on odd-sized arrays.
    """
    pad_x, pad_y = (103, 103)
    for image in ranmdom_arrays_odd.arrays:
        image_padded = pad_reflect(
            array=image,
            pad_width=(pad_x, pad_y),
        )
        image_fft = fft.rfft2(image_padded)
        # Omitting kernel bits for now, as they are not used in this test
        # Keeping them commented out for reference
        # kernel_fft = _ft_kernel(kernel, shape=image_padded.shape)

        smooth_fft = image_fft  # * kernel_fft

        smooth = fft.irfft2(smooth_fft, s=image_padded.shape)  # / kernel.sum()

        smooth_cut = smooth[
            pad_x:-pad_x,
            pad_y:-pad_y,
        ]
        assert smooth_cut.shape == image.shape, (
            f"Bad shape after irfft2 {smooth_cut.shape} != {image.shape}"
        )


def test_bane_fft(ranmdom_arrays: RandomArrays):
    gaussian_kernel_arr = gaussian_kernel(10)
    for array, loc, scale in zip(
        ranmdom_arrays.arrays, ranmdom_arrays.locs, ranmdom_arrays.scales, strict=False
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
        ranmdom_arrays.arrays, ranmdom_arrays.locs, ranmdom_arrays.scales, strict=False
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
