#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BANE: Background and Noise Estimation
...but with FFTs
"""

__author__ = ["Alec Thomson", "Tim Galvin"]
__version__ = "0.0.0"

import multiprocessing as mp
import os
from pathlib import Path
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import astropy.units as u
import numba as nb
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from numpy import fft
from radio_beam import Beam
from scipy import ndimage

from AegeanTools import BANE as bane
from AegeanTools import numba_polyfit

logging = bane.logging


@nb.njit(
    fastmath=True,
    cache=True,
)
def _ft_kernel(kernel: np.ndarray, shape: tuple) -> np.ndarray:
    """Compute the Fourier transform of a kernel

    Args:
        kernel (np.ndarray): 2D kernel
        shape (tuple): Shape of the image

    Returns:
        np.ndarray: FFT of the kernel
    """
    return fft.rfft2(kernel, s=shape)


@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.float32[:, :], nb.float32),
    fastmath=True,
    cache=True,
)
def fft_average(image: np.ndarray, kernel: np.ndarray, kern_sum: float) -> np.ndarray:
    """Compute an average with FFT magic

    Args:
        image (np.ndarray): 2D image to average spatially
        kernel (np.ndarray): 2D kernel
        kern_sum (float): Sum of the kernel in image space

    Returns:
        np.ndarray: Averaged image
    """
    image_fft = fft.rfft2(image)
    kernel_fft = _ft_kernel(kernel, shape=image.shape)

    smooth_fft = image_fft * kernel_fft

    smooth = fft.irfft2(smooth_fft) / kern_sum
    return smooth


@nb.njit(
    nb.types.UniTuple(nb.float32[:, :], 2)(
        nb.float32[:, :], nb.float32[:, :], nb.float32
    ),
    fastmath=True,
    cache=True,
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


def tophat_kernel(diameter: int):
    """Make a tophat kernel

    Args:
        radius (int): Radius of the kernel

    Returns:
        np.ndarray: Tophat kernel
    """
    radius = diameter // 2
    kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
    xx = np.arange(-radius, radius + 1)
    yy = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(xx, yy)
    mask = X**2 + Y**2 <= radius**2
    kernel[mask] = 1
    return kernel


def gaussian_kernel(fwhm: int) -> np.ndarray:
    """Make a Gaussian kernel

    Args:
        fwhm (int): FWHM of the kernel in pixels

    Returns:
        np.ndarray: Gaussian kernel
    """
    kernel = np.zeros((fwhm * 2 + 1, fwhm * 2 + 1), dtype=np.float32)
    xx = np.arange(-fwhm, fwhm + 1)
    yy = np.arange(-fwhm, fwhm + 1)
    X, Y = np.meshgrid(xx, yy)
    kernel = np.exp(-4 * np.log(2) * (X**2 + Y**2) / fwhm**2)
    return kernel.astype(np.float32)


@nb.njit(
    nb.boolean[:, :](nb.float32[:, :], nb.float32[:, :]),
    cache=True,
)
def get_nan_mask(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Get a mask of NaNs in the image

    Args:
        image (np.ndarray): Image to mask

    Returns:
        np.ndarray: Mask of NaNs
    """
    immask = np.isfinite(image)
    immask_fft = fft.rfft2(immask)
    kernel_fft = _ft_kernel(kernel, shape=image.shape)
    conv = fft.irfft2(immask_fft * kernel_fft)
    mask = conv < 1
    return mask


def get_kernel(
    header: Union[fits.Header, dict],
    step_size: Optional[int] = None,
    box_size: Optional[int] = None,
    kernel_func: Callable = gaussian_kernel,
) -> Tuple[np.ndarray, float, int]:
    """Get the kernel for FFT BANE

    Args:
        header (Union[fits.Header, dict]): Header of the image
        step_size (Optional[int], optional): Step size in pixels. Defaults to 3/beam. Values of < 0 will specify the number of pixels per beam.
        box_size (Optional[int], optional): Box size in pixels. Defaults to None. Values of < 0 will specify the number of pixels per beam.


    Returns:
        Tuple[np.ndarray, float]: The kernel and sum of the kernel
    """

    if not step_size or step_size < 0 or not box_size or box_size < 0:
        # Use the beam to determine the step/box size
        try:
            beam = Beam.from_fits_header(header)
            logging.info(f"Beam: {beam.__repr__()}")
            scales = proj_plane_pixel_scales(WCS(header)) * u.deg
            pix_per_beam = beam.minor / scales.min()
        except ValueError:
            raise ValueError(
                "Could not parse beam from header - try specifying step size"
            )

    if not step_size or step_size < 0:
        # Step size
        npix_step = 3 if not step_size else abs(step_size)
        logging.info(f"Using step size of {npix_step} pixels per beam")
        step_size = int(np.ceil(pix_per_beam / npix_step))
    logging.info(f"Using step size of {step_size} pixels")

    if not box_size or box_size < 0:
        # Box size
        npix_box = 10 if not box_size else abs(box_size)
        logging.info(f"Using a box size of {npix_box} per beam")
        box_size = int(np.ceil(pix_per_beam * npix_box / step_size))
    logging.info(f"Using box size of {box_size} pixels (scaled by step size)")

    kernel = kernel_func(box_size)
    kernel /= kernel.max()
    kern_sum = kernel.sum()

    return kernel, kern_sum, step_size


@nb.njit(
    nb.int32[:, :](nb.types.UniTuple(nb.int32, 2), nb.int32),
    fastmath=True,
    cache=True,
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


@nb.njit(
    nb.float32(
        nb.float32[:],
        nb.types.unicode_type,
        nb.int32,
        nb.float32,
        nb.float32,
    ),
    cache=True,
)
def estimate_rms(
    data: np.ndarray,
    mode: str = "mad",
    clip_rounds: int = 2,
    bin_perc: float = 0.25,
    outlier_thres: float = 3.0,
) -> float:
    """Calculates to RMS of an image, primiarily for radio interferometric images. First outlying
    pixels will be flagged. To the remaining valid pixels a Guassian distribution is fitted to the
    pixel distribution histogram, with the standard deviation being return.

    Arguments:
        data (np.ndarray) -- 1D data to estimate the noise level of

    Keyword Arguments:
        mode (str) -- Clipping mode used to flag outlying pixels, either made on the median absolute deviation (`mad`) or standard deviation (`std`) (default: ('mad'))
        clip_rounds (int) -- Number of times to perform the clipping of outlying pixels (default: (2))
        bin_perc (float) -- Bins need to have `bin_perc*MAX(BINS)` of counts to be included in the fitting procedure (default: (0.25))
        outlier_thres (float) -- Number of units of the adopted outlier statistic required for a item to be considered an outlier (default: (3))
        nan_check (bool) -- If true, non-finite values will be removed from the `data` which would otherwise cause the rms derivation to fail. If fail `data` remains untouched (default: (True))

    Raises:
        ValueError: Raised if a mode is specified but not supported

    Returns:
        float -- Estimated RMS of the supploed image
    """
    if bin_perc > 1.0:
        bin_perc /= 100.0

    if mode == "std":
        clipping_func = lambda data: np.std(data)

    elif mode == "mad":
        clipping_func = lambda data: np.median(np.abs(data - np.median(data)))

    else:
        raise ValueError(
            f"{mode} not supported as a clipping mode, available modes are `std` and `mad`. "
        )

    cen_func = lambda data: np.median(data)

    for i in range(clip_rounds):
        data = data[np.abs(data - cen_func(data)) < outlier_thres * clipping_func(data)]

    # Attempts to ensure a sane number of bins to fit against
    mask_counts = 0
    loop = 1
    while mask_counts < 5 and loop < 5:
        counts, binedges = np.histogram(data, bins=50 * loop)
        binc = (binedges[:-1] + binedges[1:]) / 2

        mask = counts >= bin_perc * np.max(counts)
        mask_counts = np.sum(mask)
        loop += 1

    # p = np.polyfit(binc[mask], np.log10(counts[mask] / np.max(counts)), 2)
    p = numba_polyfit.fit_poly(binc[mask], np.log10(counts[mask] / np.max(counts)), 2)
    a, b, c = p

    x1 = (-b + np.sqrt(b**2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    x2 = (-b - np.sqrt(b**2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    fwhm = np.abs(x1 - x2)
    noise = fwhm / 2.355

    return noise


def estimate_rms_astropy(image: np.ndarray):
    """Estimate the RMS of an image using astropy

    Args:
        image (np.ndarray): Image to estimate the RMS of

    Returns:
        float: RMS of the image
    """

    # Sigma clip the image
    clipped_image = sigma_clip(
        image,
        sigma=3,
        maxiters=None,
        cenfunc=np.nanmedian,
        stdfunc=mad_std,
        masked=False,
        copy=False,
    )
    return mad_std(clipped_image)


def robust_bane(
    image: np.ndarray,
    header: Union[fits.Header, dict],
    step_size: Optional[int] = None,
    box_size: Optional[int] = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
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
    kernel, kern_sum, step_size = get_kernel(
        header=header,
        step_size=step_size,
        box_size=box_size,
        kernel_func=kernel_func,
    )
    # nan_mask = get_nan_mask(image, kernel)
    nan_mask = ~np.isfinite(image)
    image_mask = np.nan_to_num(image)

    # Quick and dirty rms estimate
    rms_est = rms_estimator(image_mask[~nan_mask].ravel())
    snr = np.abs(image_mask) / rms_est
    mask = snr > 5
    # Clip and fill sources with noise
    image_mask[mask] = np.random.normal(
        loc=0, scale=rms_est, size=image_mask[mask].shape
    )

    # Downsample the image
    # Create slice for downsampled image
    # Ensure downsampled image has even number of pixels
    start_idx = step_size
    stop_x = image_mask.shape[1] - step_size
    stop_y = image_mask.shape[0] - step_size

    divx, modx = divmod(stop_x, step_size)
    divy, mody = divmod(stop_y, step_size)

    while divx % 2 != 0:
        stop_x -= 1
        divx, modx = divmod(stop_x, step_size)

    while divy % 2 != 0:
        stop_y -= 1
        divy, mody = divmod(stop_y, step_size)

    x_slice = slice(start_idx, stop_x, step_size)
    y_slice = slice(start_idx, stop_y, step_size)
    image_ds = image_mask[(y_slice, x_slice)]
    logging.info(f"Downsampled image to {image_ds.shape}")
    for i in range(2):
        assert (
            image_ds.shape[i] % 2 == 0
        ), "Downsampled image must have even number of pixels"

    # Create zoom factor for upsampling
    zoom_x = image.shape[1] / image_ds.shape[1]
    zoom_y = image.shape[0] / image_ds.shape[0]
    zoom = (zoom_y, zoom_x)

    # Run the FFT
    mean, avg_rms = bane_fft(image_ds, kernel, kern_sum)

    # Upsample the mean and RMS to the original image size
    mean_us = ndimage.shift(ndimage.zoom(mean, zoom, order=3, grid_mode=True), step_size*2)
    avg_rms_us = ndimage.shift(ndimage.zoom(avg_rms, zoom, order=3, grid_mode=True), step_size*2)

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
        shape = tuple(header[f"NAXIS{ii}"] for ii in range(1, header["NAXIS"] + 1))
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
    step_size: Optional[int] = None,
    box_size: Optional[int] = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
) -> Tuple[np.ndarray, np.ndarray]:
    logging.info(f"Running BANE on image {image.shape}")
    # Run BANE
    bkg, rms = robust_bane(
        image.astype(np.float32),
        header,
        step_size=step_size,
        box_size=box_size,
        kernel_func=kernel_func,
        rms_estimator=rms_estimator,
    )
    write_outputs(out_files, bkg, rms)

    return bkg, rms


def bane_3d_loop(
    plane: np.ndarray,
    idx: int,
    header: Union[fits.Header, dict],
    out_files: List[Path],
    ext: int = 0,
    step_size: Optional[int] = None,
    box_size: Optional[int] = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
):
    rms_file, bkg_file = out_files
    with fits.open(rms_file, memmap=True, mode="update") as rms_hdul, fits.open(
        bkg_file, memmap=True, mode="update"
    ) as bkg_hdul:
        rms = rms_hdul[ext].data
        bkg = bkg_hdul[ext].data
        logging.info(f"Running BANE on plane {idx}")
        bkg[idx], rms[idx] = robust_bane(
            plane.astype(np.float32),
            header,
            step_size=step_size,
            box_size=box_size,
            kernel_func=kernel_func,
            rms_estimator=rms_estimator,
        )
        rms_hdul.flush()
        bkg_hdul.flush()
    logging.info(f"Finished BANE on plane {idx}")


def bane_3d(
    cube: np.ndarray,
    header: Union[fits.Header, dict],
    out_files: List[Path],
    ext: int = 0,
    step_size: Optional[int] = None,
    box_size: Optional[int] = None,
    ncores: Optional[int] = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
) -> Tuple[np.ndarray, np.ndarray]:
    logging.info(f"Running BANE on cube {cube.shape}")
    # Run BANE
    ncores = mp.cpu_count() if not ncores else ncores
    logging.info(f"Running BANE with {ncores} cores")
    with mp.Pool(ncores) as pool:
        pool.starmap(
            bane_3d_loop,
            [
                (
                    cube[ii],
                    ii,
                    header,
                    out_files,
                    ext,
                    step_size,
                    box_size,
                    kernel_func,
                    rms_estimator,
                )
                for ii in range(cube.shape[0])
            ],
        )

    logging.info(f"Finished BANE on cube")
    rms_file, bkg_file = out_files
    with fits.open(rms_file, memmap=True, mode="denywrite") as rms_hdul, fits.open(
        bkg_file, memmap=True, mode="denywrite"
    ) as bkg_hdul:
        rms = rms_hdul[ext].data
        bkg = bkg_hdul[ext].data
    return bkg, rms


def fits_idx_to_np(
    fits_idx: int,
    header: Union[fits.Header, dict],
) -> int:
    """Convert FITS index to numpy index

    Args:
        fits_idx (int): FITS index
        header (Union[fits.Header, dict]): FITS header

    Returns:
        int: numpy index
    """
    # FITS index is 1, 2, 3, ...
    # numpy index is 0, 1, 2, ...
    # numpy index is reversed
    return header["NAXIS"] - fits_idx


def find_stokes_axis(header: Union[fits.Header, dict]) -> int:
    """Find the Stokes axis

    Args:
        header (Union[fits.Header, dict]): FITS header

    Returns:
        int: Stokes axis (numpy index)
    """
    stokes_axis = None
    for ii in range(1, header["NAXIS"] + 1):
        if header[f"CTYPE{ii}"] == "STOKES":
            stokes_axis = ii
            break
    if stokes_axis is None:
        raise ValueError("No Stokes axis found")
    return fits_idx_to_np(stokes_axis, header)


def main(
    fits_file: Path,
    ext: int = 0,
    step_size: Optional[int] = None,
    box_size: Optional[int] = None,
    ncores: Optional[int] = None,
    kernel_str: str = "gauss",
    estimator_str: str = "mad_std",
) -> Tuple[np.ndarray, np.ndarray]:
    # Init output files
    out_files = init_outputs(fits_file, ext=ext)
    # Check for frequency axis and Stokes axis
    logging.info(f"Opening FITS file {fits_file}")
    with fits.open(fits_file, memmap=True, mode="denywrite") as hdul:
        data = hdul[ext].data
        header = hdul[ext].header

    is_stokes_cube = len(data.shape) > 3 and data.shape[-1] > 1
    is_cube = len(data.shape) == 3

    estimators: Dict[str, Callable] = {
        "galvin": estimate_rms,
        "mad_std": mad_std,
        "astropy": estimate_rms_astropy,
    }
    kernels: Dict[str, Callable] = {
        "gauss": gaussian_kernel,
        "tophat": tophat_kernel,
    }

    if is_stokes_cube:
        logging.info("Detected Stokes cube")

        # Check if Stokes axis is unitary
        stokes_axis = find_stokes_axis(header)
        if data.shape[stokes_axis] != 1:
            raise NotImplementedError("Stokes cube not implemented")

        # Remove Stokes axis
        # Create slice to index all but Stokes axis
        slices = [slice(None)] * len(data.shape)
        slices[stokes_axis] = 0
        data = data[tuple(slices)]
        is_cube = True

    if is_cube:
        logging.info("Detected cube")
        bkg, rms = bane_3d(
            cube=data,
            header=header,
            out_files=out_files,
            ext=ext,
            step_size=step_size,
            box_size=box_size,
            ncores=ncores,
            kernel_func=kernels.get(kernel_str, gaussian_kernel),
            rms_estimator=estimators.get(estimator_str, mad_std),
        )

    else:
        logging.info("Detected 2D image")
        bkg, rms = bane_2d(
            image=data,
            header=header,
            out_files=out_files,
            step_size=step_size,
            box_size=box_size,
            kernel_func=kernels.get(kernel_str, gaussian_kernel),
            rms_estimator=estimators.get(estimator_str, mad_std),
        )

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
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Step size for BANE. Negative values will be interpreted as number of pixels per beam.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=None,
        help="Box size for BANE. Negative values will be interpreted as number of pixels per beam.",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=None,
        help="Number of cores to use (only sppeds up cube processing). Default is all cores.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="gauss",
        choices=["gauss", "tophat"],
        help="Kernel to use for convolution",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="mad_std",
        choices=["mad_std", "galvin", "astropy"],
        help="RMS estimator to use",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s: {__doc__} -- version {__version__}",
    )
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level, format="%(process)d:%(levelname)s %(message)s"
    )

    _ = main(
        Path(args.fits_file),
        ext=args.ext,
        step_size=args.step_size,
        box_size=args.box_size,
        ncores=args.ncores,
        kernel_str=args.kernel,
        estimator_str=args.estimator,
    )

    return 0


if __name__ == "__main__":
    cli()
