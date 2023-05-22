#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BANE: Background and Noise Estimation
...but with FFTs
"""

import os
import multiprocessing as mp
from pathlib import Path
from time import time
from typing import List, Tuple, Union

import numba as nb
import numpy as np
from astropy.io import fits
from scipy import interpolate, ndimage

from AegeanTools import BANE as bane

logging = bane.logging

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


@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.float32[:, :], nb.float32),
    fastmath=True,
    parallel=True,
)
def fft_average(
    image: np.ndarray, kernel: np.ndarray, kern_sum: float
) -> np.ndarray:
    """Compute an average with FFT magic

    Args:
        image (np.ndarray): 2D image to average spatially
        kernel (np.ndarray): 2D kernel
        kern_sum (float): Sum of the kernel in image space

    Returns:
        np.ndarray: Averaged image
    """
    image_fft = np.fft.rfft2(image)
    kernel_fft = _ft_kernel(kernel, shape=image.shape)

    smooth_fft = image_fft * kernel_fft

    smooth = np.fft.irfft2(smooth_fft) / kern_sum
    return smooth


@nb.njit(
    nb.types.UniTuple(
        nb.float32[:, :],
        2
    )(nb.float32[:, :], nb.float32[:, :], nb.float32),
    fastmath=True,
    parallel=True,
)
def bane_fft(
    image: np.ndarray,
    kernel: np.ndarray,
    kern_sum: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """BANE but with FFTs

    Args:
        image (np.ndarray): Image to find background and RMS of
        kernel (np.ndarray): Tophat kernel
        kern_sum (float): Sum of the kernel in image domain

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and RMS of the image
    """
    mean = fft_average(image, kernel, kern_sum)
    rms = np.sqrt((image - mean) ** 2)
    avg_rms = fft_average(rms, kernel, kern_sum)
    return mean, avg_rms


def tophat_kernel(radius: int):
    """Make a tophat kernel

    Args:
        radius (int): Radius of the kernel

    Returns:
        np.ndarray: Tophat kernel
    """
    kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
    xx = np.arange(-radius, radius + 1)
    yy = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(xx, yy)
    mask = X ** 2 + Y ** 2 <= radius ** 2
    kernel[mask] = 1
    return kernel


def get_kernel(header: Union[fits.Header, dict]) -> Tuple[np.ndarray, float, int]:
    """Get the kernel for FFT BANE

    Args:
        header (Union[fits.Header, dict]): Header of the image

    Returns:
        Tuple[np.ndarray, float]: The kernel and sum of the kernel
    """
    step_size = max(bane.get_step_size(header))
    box_size = step_size * 6
    kernel = tophat_kernel(radius=box_size // 2)
    kernel /= kernel.max()

    kern_sum = kernel.sum()

    return kernel, kern_sum, step_size

@nb.njit(
    nb.int32[:, :](nb.types.UniTuple(nb.int32, 2), nb.int32),
    fastmath=True,
    parallel=True,
)
def chunk_image(image_shape: Tuple[int, int], box_size: int) -> np.ndarray:
    """Divide the image into chunks that overlap by half the box size

    Chunk only the y-axis

    Args:
        image_shape (Tuple[int, int]): Shape of the image
        box_size (int): Size of the box

    Returns:
        np.ndarray: Chunk coordinates (start, end) x nchunks
    """

    nchunks = image_shape[0] // (box_size // 2) - 1

    chunks = np.zeros((nchunks, 2), dtype=np.int32)

    for i in nb.prange(nchunks):
        chunks[i] = [i * (box_size // 2), (i + 2) * (box_size // 2)]

    chunks[-1, 1] = image_shape[0]

    return chunks


def robust_bane(
    image: np.ndarray, header: Union[fits.Header, dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-round BANE with FFTs

    Args:
        image (np.ndarray): Image to find background and RMS of
        header (Union[fits.Header, dict]): Header of the image

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and RMS of the image
    """
    logging.info("Running FFT BANE")
    tick = time()
    # Setups
    kernel, kern_sum, step_size = get_kernel(header)
    nan_mask = ~np.isfinite(image)
    image_mask = np.nan_to_num(image)
    
    # Downsample the image
    image_ds = image_mask[::step_size, ::step_size]

    # Round 1
    mean, avg_rms = bane_fft(image_ds, kernel, kern_sum)
    # Round 2
    # Repeat with masked values filled in with noise
    snr = np.abs(image_ds) / avg_rms
    mask = snr > 5
    image_masked = image_ds.copy()
    image_masked[mask] = np.random.normal(
        loc=0, scale=avg_rms.mean(), size=image_masked[mask].shape
    )
    mean, avg_rms = bane_fft(image_masked, kernel, kern_sum)

    # Upsample the mean and RMS to the original image size
    zoom_x = image.shape[1] / image_ds.shape[1]
    zoom_y = image.shape[0] / image_ds.shape[0]
    zoom = (zoom_y, zoom_x)
    mean_us = ndimage.zoom(mean, zoom, order=1)
    avg_rms_us = ndimage.zoom(avg_rms, zoom, order=1)

    # Reapply mask
    mean_us[nan_mask] = np.nan
    avg_rms_us[nan_mask] = np.nan

    tock = time()

    logging.info(f"FFT BANE took {tock - tick:.2f} seconds")

    return mean_us, avg_rms_us

def init_outputs(
    fits_file: Path,
    ext: int = 0,
) -> List[Path]:
    """Initialize the output files

    Args:
        fits_file (Path): Input FITS file
        ext (int, optional): HDU extension. Defaults to 0.
    """    
    logging.info("Initializing output files")
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

        logging.info(f"Created {out_file}")
        out_files.append(out_file)

    return out_files

def write_outputs(
        out_files: List[Path],
        mean: np.ndarray,
        rms: np.ndarray,
):
    rms_file, bkg_file = out_files
    with fits.open(rms_file, memmap=True, mode="update") as hdul:
        logging.info(f"Writing RMS to {rms_file}")
        hdul[0].data = rms
        hdul.flush()
    logging.info(f"Wrote RMS to {rms_file}")

    with fits.open(bkg_file, memmap=True, mode="update") as hdul:
        logging.info(f"Writing background to {bkg_file}")
        hdul[0].data = mean
        hdul.flush()
    logging.info(f"Wrote background to {bkg_file}")


def bane_2d(
        image: np.ndarray,
        header: Union[fits.Header, dict],
        out_files: List[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    logging.info(f"Running BANE on image {image.shape}")
    # Run BANE
    bkg, rms = robust_bane(image, header)
    write_outputs(out_files, bkg, rms)

    return bkg, rms


def bane_3d_loop(
    plane: np.ndarray,
    idx: int,
    header: Union[fits.Header, dict],
    out_files: List[Path],
    ext: int = 0,
):
    rms_file, bkg_file = out_files
    with fits.open(
        rms_file, memmap=True, mode="update"
    ) as rms_hdul, fits.open(
        bkg_file, memmap=True, mode="update"
    ) as bkg_hdul:
        rms = rms_hdul[ext].data
        bkg = bkg_hdul[ext].data
        logging.info(f"Running BANE on plane {idx}")
        bkg[idx], rms[idx] = robust_bane(plane, header)
        rms_hdul.flush()
        bkg_hdul.flush()
    logging.info(f"Finished BANE on plane {idx}")

def bane_3d(
        cube: np.ndarray,
        header: Union[fits.Header, dict],
        out_files: List[Path],
        ext: int = 0,
):
    logging.info(f"Running BANE on cube {cube.shape}")
    # Run BANE
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(
            bane_3d_loop,
            [
                (cube[ii], ii, header, out_files, ext)
                for ii in range(cube.shape[0])
            ],
        )       
        
    logging.info(f"Finished BANE on cube")
    rms_file, bkg_file = out_files
    with fits.open(
        rms_file, memmap=True, mode="denywrite"
    ) as rms_hdul, fits.open(
        bkg_file, memmap=True, mode="denywrite"
    ) as bkg_hdul:
        rms = rms_hdul[ext].data
        bkg = bkg_hdul[ext].data
    return bkg, rms

def main(
    fits_file: Path,
    ext: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    # Init output files
    out_files = init_outputs(fits_file, ext=ext)
    # Check for frequency axis and Stokes axis
    logging.info(f"Opening FITS file {fits_file}")
    with fits.open(fits_file, memmap=True, mode="denywrite") as hdul:
        data = hdul[ext].data.astype(np.float32)
        header = hdul[ext].header
        
    is_stokes_cube = len(data.shape) > 3 and data.shape[-1] > 1
    is_cube = len(data.shape) == 3

    from IPython import embed
    embed()

    if is_stokes_cube:
        logging.info("Detected Stokes cube")
        raise NotImplementedError("Stokes cube not implemented")


    if is_cube:
        logging.info("Detected cube")
        bkg, rms = bane_3d(data, header, out_files, ext=ext)

        
    else:
        logging.info("Detected 2D image")
        bkg, rms = bane_2d(data, header, out_files)


    logging.info("Done")

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level,
                        format="%(process)d:%(levelname)s %(message)s")

    _ = main(
        Path(args.fits_file),
        ext=args.ext,
    )

    return 0

if __name__ == "__main__":
    cli()