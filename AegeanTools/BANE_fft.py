"""
BANE: Background and Noise Estimation
...but with FFTs

Unlike the original BANE, this version uses FFTs to compute the background and noise.
Further, the `step` and `box` parameters are now the downsampling factor and kernel size, respectively.
Downsampling is done by taking every `step`th pixel in each dimension, and the kernel is applied to the downsampled image.

"""

from __future__ import annotations

# TODO: Images come out with a slight offset. Need to figure out why.

__author__ = ["Alec Thomson", "Tim Galvin"]

import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import time
from typing import Any

import astropy.units as u
import numba as nb
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from numpy import fft
from numpy.typing import NDArray
from radio_beam import Beam
from scipy import ndimage

from AegeanTools import numba_polyfit

logging.basicConfig(
    format="%(module)s:%(levelname)s %(message)s",
    level=logging.INFO,
)

rng = np.random.default_rng()


@nb.njit(
    fastmath=True,
    cache=True,
)
def _ft_kernel(kernel: NDArray[np.float32], shape: tuple) -> NDArray[np.float32]:
    """Compute the Fourier transform of a kernel

    Args:
        kernel (NDArray[np.float32]): 2D kernel
        shape (tuple): Shape of the image

    Returns:
        NDArray[np.float32]: FFT of the kernel
    """
    return fft.rfft2(kernel, s=shape)


@nb.njit(
    nb.float32[:, :](
        nb.float32[:, :],
        nb.types.UniTuple(nb.int64, 2),
    ),
    fastmath=True,
    cache=True,
)
def pad_reflect(
    array: NDArray[np.float32],
    pad_width: tuple[int, int],
) -> NDArray[np.float32]:
    """Numba compatible version of np.pad with reflect mode

    Args:
        array (NDArray[np.float32]): Array to pad
        pad_width (tuple[int, int]): Width of the padding

    Raises:
        ValueError: If mode is not supported

    Returns:
        NDArray[np.float32]: Padded array
    """
    nx, ny = array.shape
    px, py = pad_width

    # Create the padded array
    padded = np.empty((nx + 2 * px, ny + 2 * py), dtype=array.dtype)

    # Copy the original array into the center
    padded[px : px + nx, py : py + ny] = array

    # Reflect top and bottom
    for i in range(px):
        padded[px - 1 - i, py : py + ny] = array[i + 1, :]
        padded[nx + px + i, py : py + ny] = array[nx - 2 - i, :]

    # Reflect left and right
    for j in range(py):
        padded[:, py - 1 - j] = padded[:, py + j + 1]
        padded[:, ny + py + j] = padded[:, ny + py - 2 - j]

    return padded


@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.float32[:, :]),
    fastmath=True,
    cache=True,
)
def fft_average(
    image: NDArray[np.float32], kernel: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Compute an average with FFT magic

    Args:
        image (NDArray[np.float32]): 2D image to average spatially
        kernel (NDArray[np.float32]): 2D kernel

    Returns:
        NDArray[np.float32]: Averaged image
    """
    # pad the image by the kernel size * 2
    pad_x, pad_y = kernel.shape
    image_padded = pad_reflect(
        array=image,
        pad_width=(pad_x, pad_y),
    )
    image_fft = fft.rfft2(image_padded)
    kernel_fft = _ft_kernel(kernel, shape=image_padded.shape)

    smooth_fft = image_fft * kernel_fft

    smooth = fft.irfft2(smooth_fft, s=image_padded.shape) / kernel.sum()

    smooth_cut = smooth[
        pad_x:-pad_x,
        pad_y:-pad_y,
    ]
    assert smooth_cut.shape == image.shape
    return smooth_cut


@nb.njit(
    nb.types.UniTuple(nb.float32[:, :], 2)(nb.float32[:, :], nb.float32[:, :]),
    fastmath=True,
    cache=True,
)
def bane_fft(
    image: NDArray[np.float32],
    kernel: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """BANE but with FFTs

    Args:
        image (NDArray[np.float32]): Image to find background and RMS of
        kernel (NDArray[np.float32]): Tophat kernel

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.float32]]: Mean and RMS of the image
    """
    mean = fft_average(image, kernel)
    rms = (image - mean) ** 2
    avg_rms = np.sqrt(fft_average(rms, kernel))
    return mean, avg_rms


def tophat_kernel(diameter: int):
    """Make a tophat kernel

    Args:
        radius (int): Radius of the kernel

    Returns:
        NDArray[np.float32]: Tophat kernel
    """
    radius = diameter // 2
    kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.float32)
    xx = np.arange(-radius, radius + 1)
    yy = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(xx, yy)
    mask = radius**2 >= X**2 + Y**2
    kernel[mask] = 1
    return kernel


def gaussian_kernel(fwhm: int) -> NDArray[np.float32]:
    """Make a Gaussian kernel

    Args:
        fwhm (int): FWHM of the kernel in pixels

    Returns:
        NDArray[np.float32]: Gaussian kernel
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
def get_nan_mask(
    image: NDArray[np.float32], kernel: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Get a mask of NaNs in the image

    Args:
        image (NDArray[np.float32]): Image to mask

    Returns:
        NDArray[np.float32]: Mask of NaNs
    """
    immask = np.isfinite(image)
    immask_fft = fft.rfft2(immask)
    kernel_fft = _ft_kernel(kernel, shape=image.shape)
    conv = fft.irfft2(immask_fft * kernel_fft)
    return conv < 1


def get_kernel(
    header: fits.Header | dict[str, Any],
    step_size: int | None = None,
    box_size: int | None = None,
    kernel_func: Callable = gaussian_kernel,
) -> tuple[NDArray[np.float32], int]:
    """Get the kernel for FFT BANE

    Note that here the `step` is the downsampling factor, and the `box` is the kernel size.

    Args:
        header (fits.Header | dict[str, Any]): Header of the image
        step_size (int | None, optional): Step size in pixels. Defaults to 3 beams. Values of < 0 will specify the number of beams/step.
        box_size (int | None, optional): Box size in pixels. Defaults to 10 beams. Values of < 0 will specify the number of beams/box.


    Returns:
        Tuple[NDArray[np.float32], float]: The kernel and sum of the kernel
    """

    logging.info(f"{step_size=}, {box_size=}")
    if step_size is None or step_size < 0 or box_size is None or box_size < 0:
        # Use the beam to determine the step/box size
        try:
            beam = Beam.from_fits_header(header)
            logging.info(f"Beam: {beam.__repr__()}")
            scales = proj_plane_pixel_scales(WCS(header)) * u.deg / u.pixel
            pix_per_beam = beam.minor / scales.min()
            logging.info(f"Pixels per beam: {pix_per_beam:0.1f}")
        except ValueError:
            msg = "Could not parse beam from header - try specifying step size"
            raise ValueError(msg)

    if step_size is None or step_size < 0:
        # Step size
        nbeam_step = 3 if step_size is None else abs(step_size)
        logging.info(f"Using step size of {nbeam_step} beams per step")
        step_size_pix = int(np.ceil((nbeam_step * pix_per_beam).to(u.pix).value))

    else:
        step_size_pix = step_size

    logging.info(f"Using step size of {step_size_pix} pixels")

    if box_size is None or box_size < 0:
        # Box size
        nbeam_box = 10 if box_size is None else abs(box_size)
        logging.info(f"Using a box size of {nbeam_box} beams per box")
        scaler = step_size_pix if step_size_pix > 0 else 1
        box_size_pix = abs(int(np.ceil(pix_per_beam.value * nbeam_box / scaler)))

    else:
        box_size_pix = box_size

    logging.info(f"Using box size of {box_size_pix} pixels (scaled by step size)")

    kernel = kernel_func(box_size_pix)
    kernel /= kernel.max()

    return kernel, step_size_pix


@nb.njit(
    nb.int32[:, :](nb.types.UniTuple(nb.int32, 2), nb.int32),
    fastmath=True,
    cache=True,
)
def chunk_image(image_shape: tuple[int, int], box_size: int) -> NDArray[np.float32]:
    """Divide the image into chunks that overlap by half the box size

    Chunk only the y-axis

    Args:
        image_shape (Tuple[int, int]): Shape of the image
        box_size (int): Size of the box

    Returns:
        NDArray[np.float32]: Chunk coordinates (start, end) x nchunks
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
    ),
    cache=True,
)
def median_jit(data: NDArray[np.float32]) -> float:
    """Median of an array

    Args:
        data (NDArray[np.float32]): Data to find the median of

    Returns:
        float: Median of the data
    """
    return np.median(data)


@nb.njit(
    nb.float32(
        nb.float32[:],
    ),
    cache=True,
)
def std_jit(data: NDArray[np.float32]) -> float:
    """Standard deviation of an array

    Args:
        data (NDArray[np.float32]): Data to find the standard deviation of

    Returns:
        float: Standard deviation of the data
    """
    return np.std(data)


@nb.njit(
    nb.float32(
        nb.float32[:],
    ),
    cache=True,
)
def mad_jit(data: NDArray[np.float32]) -> float:
    """Median absolute deviation of an array

    Args:
        data (NDArray[np.float32]): Data to find the median absolute deviation of

    Returns:
        float: Median absolute deviation of the data
    """
    return np.median(np.abs(data - np.median(data)))


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
    data: NDArray[np.float32],
    mode: str = "mad",
    clip_rounds: int = 2,
    bin_perc: float = 0.25,
    outlier_thres: float = 3.0,
) -> float:
    """Calculates to RMS of an image, primiarily for radio interferometric images. First outlying
    pixels will be flagged. To the remaining valid pixels a Guassian distribution is fitted to the
    pixel distribution histogram, with the standard deviation being return.

    Arguments:
        data (NDArray[np.float32]) -- 1D data to estimate the noise level of

    Keyword Arguments:
        mode (str) -- Clipping mode used to flag outlying pixels, either made on the median absolute deviation (`mad`) or standard deviation (`std`) (default: ('mad'))
        clip_rounds (int) -- Number of times to perform the clipping of outlying pixels (default: (2))
        bin_perc (float) -- Bins need to have `bin_perc*MAX(BINS)` of counts to be included in the fitting procedure (default: (0.25))
        outlier_thres (float) -- Number of units of the adopted outlier statistic required for a item to be considered an outlier (default: (3))

    Raises:
        ValueError: Raised if a mode is specified but not supported

    Returns:
        float -- Estimated RMS of the supploed image
    """
    if bin_perc > 1.0:
        bin_perc /= 100.0

    if mode == "std":
        clipping_func = std_jit

    elif mode == "mad":
        clipping_func = mad_jit

    else:
        msg = f"{mode} not supported as a clipping mode, available modes are `std` and `mad`. "
        raise ValueError(msg)

    cen_func = median_jit

    for _i in range(clip_rounds):
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
    return fwhm / 2.355


def estimate_rms_astropy(image: NDArray[np.float32]):
    """Estimate the RMS of an image using astropy

    Args:
        image (NDArray[np.float32]): Image to estimate the RMS of

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
    image: NDArray[np.float32],
    header: fits.Header | dict[str, Any],
    step_size: int | None = None,
    box_size: int | None = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
    clip_sigma: float = 5,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Two-round BANE with FFTs

    Note that here the `step` is the downsampling factor, and the `box` is the kernel size.
    The first round is a quick RMS estimate, the second round is the actual BANE.

    A round of clipping is done to remove sources from the image. The clipped image is then filled with
    noise drawn from a Gaussian distribution with the estimated RMS. The image is then downsampled by the
    step size, and the kernel is applied to the image. The kernel is then upsampled back to the original
    image size.

    Args:
        image (NDArray[np.float32]): Image to find background and RMS of
        header (fits.Header | dict[str, Any]): Header of the image
        step_size (int | None, optional): Step size in pixels. Defaults to 3 beams. Values of < 0 will specify the number of beams/step.
        box_size (int | None, optional): Box size in pixels. Defaults to 10 beams. Values of < 0 will specify the number of beams/box.
        kernel_func (Callable, optional): Kernel function to use. Defaults to gaussian_kernel.
        rms_estimator (Callable, optional): RMS estimator to use. Defaults to mad_std.
        clip_sigma (float, optional): Sigma to clip the image. Defaults to 5.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.float32]]: Mean and RMS of the image
    """
    logging.info("Running FFT BANE")
    tick = time()
    # Setups
    kernel, step_size_pix = get_kernel(
        header=header,
        step_size=step_size,
        box_size=box_size,
        kernel_func=kernel_func,
    )
    assert step_size_pix >= 0, "Step size must be positive"
    # nan_mask = get_nan_mask(image, kernel)
    nan_mask = ~np.isfinite(image)
    image_mask = np.nan_to_num(image)

    # Quick and dirty rms estimate
    rms_est = rms_estimator(image_mask[~nan_mask].ravel())
    snr = np.abs(image_mask) / rms_est
    mask = snr > clip_sigma
    logging.info(f"Quick RMS estimate: {rms_est:.2f}")
    logging.info(
        f"Masking {np.sum(mask)} ({np.sum(mask) / image.size * 100:0.1f}%) pixels with SNR > {clip_sigma}"
    )
    # Clip and fill sources with noise
    image_mask[mask] = rng.normal(loc=0, scale=rms_est, size=image_mask[mask].shape)
    if step_size_pix > 0:
        logging.info(f"Downsampling image by {step_size_pix} pixels")
        # Downsample the image
        # Create slice for downsampled image
        # Ensure downsampled image has even number of pixels
        start_idx = step_size_pix
        stop_x = image_mask.shape[1] - step_size_pix
        stop_y = image_mask.shape[0] - step_size_pix

        divx, modx = divmod(stop_x, step_size_pix)
        divy, mody = divmod(stop_y, step_size_pix)

        while divx % 2 != 0:
            stop_x -= 1
            divx, modx = divmod(stop_x, step_size_pix)

        while divy % 2 != 0:
            stop_y -= 1
            divy, mody = divmod(stop_y, step_size_pix)

        x_slice = slice(start_idx, stop_x, step_size_pix)
        y_slice = slice(start_idx, stop_y, step_size_pix)
        image_mask = image_mask[(y_slice, x_slice)]
        logging.info(f"Downsampled image to {image_mask.shape}")

        # Create zoom factor for upsampling
        zoom_x = image.shape[1] / image_mask.shape[1]
        zoom_y = image.shape[0] / image_mask.shape[0]
        zoom = (zoom_y, zoom_x)

    # Run the FFT
    mean, avg_rms = bane_fft(image_mask, kernel)
    # Catch small values
    mean = np.nan_to_num(mean, nan=0.0)
    avg_rms = np.nan_to_num(avg_rms, nan=0.0)

    if step_size_pix > 0:
        logging.info("Upsampling back to original image size")
        # Upsample the mean and RMS to the original image size
        # Trying a shift first to see if it helps with the edge effects
        # mean_shift = ndimage.shift(mean, box_size*step_size)
        # avg_rms_shift = ndimage.shift(avg_rms, box_size*step_size)
        mean = ndimage.zoom(mean, zoom, order=3, grid_mode=True, mode="reflect")
        avg_rms = ndimage.zoom(avg_rms, zoom, order=3, grid_mode=True, mode="reflect")

    # Reapply mask
    mean[nan_mask] = np.nan
    avg_rms[nan_mask] = np.nan

    tock = time()

    logging.info(f"FFT BANE took {tock - tick:.2f} seconds")

    return mean, avg_rms


def init_outputs(
    fits_file: Path,
    ext: int = 0,
) -> list[Path]:
    """Initialize the output files

    Args:
        fits_file (Path): Input FITS file
        ext (int, optional): HDU extension. Defaults to 0.
    """
    logging.info("Initializing output files")
    out_files: list[Path] = []
    with fits.open(fits_file, memmap=True, mode="denywrite") as hdul:
        header = hdul[ext].header
    # Create an arbitrarly large file without holding it in memory
    for suffix in ("rms", "bkg"):
        out_file = Path(fits_file.as_posix().replace(".fits", f"_{suffix}.fits"))
        if out_file.exists():
            logging.warning(f"Removing existing {out_file}")
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
    out_files: list[Path],
    mean: NDArray[np.float32],
    rms: NDArray[np.float32],
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
    image: NDArray[np.float32],
    header: fits.Header | dict[str, Any],
    out_files: list[Path],
    step_size: int | None = None,
    box_size: int | None = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
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
    plane: NDArray[np.float32],
    idx: int,
    header: fits.Header | dict[str, Any],
    out_files: list[Path],
    ext: int = 0,
    step_size: int | None = None,
    box_size: int | None = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
):
    rms_file, bkg_file = out_files
    with (
        fits.open(rms_file, memmap=True, mode="update") as rms_hdul,
        fits.open(bkg_file, memmap=True, mode="update") as bkg_hdul,
    ):
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
    cube: NDArray[np.float32],
    header: fits.Header | dict[str, Any],
    out_files: list[Path],
    ext: int = 0,
    step_size: int | None = None,
    box_size: int | None = None,
    ncores: int | None = None,
    kernel_func: Callable = gaussian_kernel,
    rms_estimator: Callable = mad_std,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    logging.info(f"Running BANE on cube {cube.shape}")
    # Run BANE
    ncores = ncores if ncores else mp.cpu_count()
    logging.info(f"Running BANE with {ncores} cores")
    with ThreadPool(ncores) as pool:
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

    logging.info("Finished BANE on cube")
    rms_file, bkg_file = out_files
    with (
        fits.open(rms_file, memmap=True, mode="denywrite") as rms_hdul,
        fits.open(bkg_file, memmap=True, mode="denywrite") as bkg_hdul,
    ):
        rms = rms_hdul[ext].data
        bkg = bkg_hdul[ext].data
    return bkg, rms


def fits_idx_to_np(
    fits_idx: int,
    header: fits.Header | dict[str, Any],
) -> int:
    """Convert FITS index to numpy index

    Args:
        fits_idx (int): FITS index
        header (fits.Header | dict[str, Any]): FITS header

    Returns:
        int: numpy index
    """
    # FITS index is 1, 2, 3, ...
    # numpy index is 0, 1, 2, ...
    # numpy index is reversed
    return header["NAXIS"] - fits_idx


def find_stokes_axis(header: fits.Header | dict[str, Any]) -> int:
    """Find the Stokes axis

    Args:
        header (fits.Header | dict[str, Any]): FITS header

    Returns:
        int: Stokes axis (numpy index)
    """
    stokes_axis = None
    for ii in range(1, header["NAXIS"] + 1):
        if header[f"CTYPE{ii}"] == "STOKES":
            stokes_axis = ii
            break
    if stokes_axis is None:
        msg = "No Stokes axis found"
        raise ValueError(msg)
    return fits_idx_to_np(stokes_axis, header)


def main(
    fits_file: Path,
    ext: int = 0,
    step_size: int | None = None,
    box_size: int | None = None,
    ncores: int | None = None,
    kernel_str: str = "gauss",
    estimator_str: str = "mad_std",
    all_in_mem: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    logging.info("Starting BANE (tools will be compiled...)")
    # Init output files
    out_files = init_outputs(fits_file, ext=ext)
    # Check for frequency axis and Stokes axis
    logging.info(f"Opening FITS file {fits_file}")
    with fits.open(fits_file, memmap=True, mode="denywrite") as hdul:
        data = hdul[ext].data.astype(np.float32)
        header = hdul[ext].header

    if all_in_mem:
        logging.warning("Loading entire image into memory!")
        data = np.array(data, dtype=np.float32)

    is_stokes_cube = len(data.shape) > 3 and data.shape[-1] > 1
    is_cube = len(data.shape) == 3

    estimators: dict[str, Callable] = {
        "galvin": estimate_rms,
        "mad_std": mad_std,
        "astropy": estimate_rms_astropy,
    }
    kernels: dict[str, Callable] = {
        "gauss": gaussian_kernel,
        "tophat": tophat_kernel,
    }

    if is_stokes_cube:
        logging.info("Detected Stokes cube")

        # Check if Stokes axis is unitary
        stokes_axis = find_stokes_axis(header)
        if data.shape[stokes_axis] != 1:
            msg = "Stokes cube not implemented"
            raise NotImplementedError(msg)

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
        help="Step size for BANE (i.e. downsampling factor). Negative values will be interpreted as number of beams per step. Set to 0 for no downsampling.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=None,
        help="Box size for BANE (i.e. kernel size). Negative values will be interpreted as number of beams per step.",
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
        "--all-in-mem",
        action="store_true",
        help="Load entire image into memory",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s: {__doc__}",
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
        all_in_mem=args.all_in_mem,
    )

    return 0


if __name__ == "__main__":
    cli()
