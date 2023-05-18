#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BANE: Background and Noise Estimation
...but with FFTs
"""

from pathlib import Path
from typing import Tuple, List
import os
from time import time
import logging

import numba as nb
import numpy as np
from astropy.convolution import Tophat2DKernel
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import StokesSpectralCube
from spectral_cube.utils import FITSReadError

from AegeanTools import BANE as bane

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.complex64[:, :], nb.float32),
    fastmath=True,
    parallel=True,
)
def fft_average(
    image: np.ndarray, kernel_fft: np.ndarray, kern_sum: float
) -> np.ndarray:
    """Compute an average with FFT magic

    Args:
        image (np.ndarray): 2D image to average spatially
        kernel_fft (np.ndarray): 2D kernel in Fourier space
        kern_sum (float): Sum of the kernel in image space

    Returns:
        np.ndarray: Averaged image
    """
    image_fft = np.fft.rfft2(image)

    smooth_fft = image_fft * kernel_fft

    smooth = np.fft.irfft2(smooth_fft) / kern_sum
    return smooth


@nb.njit(
    nb.types.UniTuple(
        nb.float32[:, :],
        2
    )(nb.float32[:, :], nb.complex64[:, :], nb.float32),
    fastmath=True,
    parallel=True,
)
def bane_fft(
    image: np.ndarray,
    kernel_fft: np.ndarray,
    kern_sum: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """BANE but with FFTs

    Args:
        image (np.ndarray): Image to find background and RMS of
        kernel_fft (np.ndarray): Tophat kernel in Fourier domain
        kern_sum (float): Sum of the kernel in image domain

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and RMS of the image
    """
    mean = fft_average(image, kernel_fft, kern_sum)
    rms = np.sqrt((image - mean) ** 2)
    avg_rms = fft_average(rms, kernel_fft, kern_sum)
    return mean, avg_rms


@nb.njit(
    fastmath=True,
    # parallel=True,
)
def _ft_kernel(kernel: np.ndarray, shape: tuple) -> np.ndarray:
    """Compute the Fourier transform of a kernel

    Args:
        kernel (np.ndarray): 2D kernel
        shape (tuple): Shape of the image

    Returns:
        np.ndarray: FFT of the kernel
    """
    return np.fft.rfft2(kernel, s=shape)


def get_kernel(header: fits.Header) -> Tuple[np.ndarray, float]:
    """Get the kernel for FFT BANE

    Args:
        header (fits.Header): Header of the image

    Returns:
        Tuple[np.ndarray, float]: FFT of the kernel and sum of the kernel
    """
    step_size = bane.get_step_size(header)
    kernel = Tophat2DKernel(radius=step_size[0] // 2 * 5).array.astype(np.float32)
    kernel /= kernel.max()
    wcs = WCS(header)

    kernel_fft = _ft_kernel(kernel, shape=wcs.celestial.array_shape)
    kern_sum = kernel.sum()

    return kernel_fft, kern_sum


def robust_bane(
    image: np.ndarray, header: fits.Header
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-round BANE with FFTs

    Args:
        image (np.ndarray): Image to find background and RMS of
        header (fits.Header): Header of the image

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and RMS of the image
    """
    logger.info("Running FFT BANE")
    tick = time()
    # Setups
    kernel_fft, kern_sum = get_kernel(header)
    nan_mask = ~np.isfinite(image)
    image = np.nan_to_num(image)

    # Round 1
    mean, avg_rms = bane_fft(image, kernel_fft, kern_sum)
    # Round 2
    # Repeat with masked values filled in with noise
    snr = np.abs(image) / avg_rms
    mask = snr > 5
    image_masked = image.copy()
    image_masked[mask] = np.random.normal(
        loc=0, scale=avg_rms.mean(), size=image_masked[mask].shape
    )
    # image_masked = inpaint.inpaint_biharmonic(image, mask)
    mean, avg_rms = bane_fft(image_masked, kernel_fft, kern_sum)

    # Reapply mask
    mean[nan_mask] = np.nan
    avg_rms[nan_mask] = np.nan

    tock = time()

    logger.info(f"FFT BANE took {tock - tick:.2f} seconds")

    return mean, avg_rms

def init_outputs(
    fits_file: Path,
    ext: int = 0,
) -> List[Path]:
    """Initialize the output files

    Args:
        fits_file (Path): Input FITS file
        ext (int, optional): HDU extension. Defaults to 0.
    """    
    logger.info("Initializing output files")
    out_files: List[Path] = []
    with fits.open(fits_file, memmap=True, mode="denywrite") as hdul:
        header = hdul[ext].header
    # Create an arbitrarly large file without holding it in memory
    for suffix in ("rms", "bkg"):
        out_file = Path(fits_file.as_posix().replace(".fits", f"_{suffix}.fits"))
        if out_file.exists():
            os.remove(out_file)

        header.tofile(out_file)
        shape = tuple(
            header[f"NAXIS{ii}"] for ii in range(1, header["NAXIS"] + 1)
        )
        with open(out_file, "rb+") as fobj:
            fobj.seek(
                len(header.tostring())
                + (np.prod(shape) * np.abs(header["BITPIX"] // 8))
                - 1
            )
            fobj.write(b"\0")

        logger.info(f"Created {out_file}")
        out_files.append(out_file)

    return out_files

def write_outputs(
        out_files: List[Path],
        mean: np.ndarray,
        rms: np.ndarray,
):
    rms_file, bkg_file = out_files
    with fits.open(rms_file, memmap=True, mode="update") as hdul:
        logger.info(f"Writing RMS to {rms_file}")
        hdul[0].data = rms
        hdul.flush()
    logger.info(f"Wrote RMS to {rms_file}")

    with fits.open(bkg_file, memmap=True, mode="update") as hdul:
        logger.info(f"Writing background to {bkg_file}")
        hdul[0].data = mean
        hdul.flush()
    logger.info(f"Wrote background to {bkg_file}")


def main(
    fits_file: Path,
    ext: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    # Init output files
    out_files = init_outputs(fits_file, ext=ext)
    # Check for frequency axis and Stokes axis
    try:
        cube = StokesSpectralCube.read(fits_file)
        is_stokes_cube = len(cube.shape) > 3 and cube.shape[-1] > 1
        if is_stokes_cube:
            logger.info("Detected Stokes cube")
            raise NotImplementedError("Stokes cube not implemented")
        
    except FITSReadError:
        logger.info("Input FITS file is an image")
        with fits.open(fits_file, memmap=True, mode="denywrite") as hdul:
            image = hdul[ext].data.astype(np.float32)
            header = hdul[ext].header
        
        assert len(image.shape) == 2, "Input image must be 2D"

        logger.info(f"Running BANE on image ({image.shape})")

        # Run BANE
        bkg, rms = robust_bane(image, header)

    finally:
        # Write output
        write_outputs(out_files, bkg, rms)

        logger.info("Done")

    return bkg, rms

def cli():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "fits_file",
        type=str,
        help="Input FITS file",
    )
    parser.add_argument(
        "--ext",
        type=int,
        default=0,
        help="HDU extension",
    )
    args = parser.parse_args()

    _ = main(
        Path(args.fits_file),
        ext=args.ext,
    )

    return 0

if __name__ == "__main__":
    cli()