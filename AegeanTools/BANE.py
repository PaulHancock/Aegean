#! /usr/bin/env python

"""
This module contains all of the BANE specific code
The function filter_image should be imported from elsewhere and run as is.
"""

import copy
import multiprocessing
import os
import platform
import sys
import uuid
from multiprocessing.shared_memory import SharedMemory
from time import gmtime, strftime

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

from AegeanTools.logging import logger

from .fits_tools import compress


# don't freak out if numba isn't installed
try:
    from numba import njit
except ImportError:

    def njit(f):
        return f


__author__ = "Paul Hancock"
__version__ = "v1.10.1"
__date__ = "2024-12-09"

# global barrier for multiprocessing
barrier = None
memory_id = None


def init(b, mem):
    """
    Set the global barrier and memory_id
    """
    global barrier, memory_id
    barrier = b
    memory_id = mem


@njit
def sigmaclip(arr, lo, hi, reps=10):
    """
    Perform sigma clipping on an array, ignoring non finite values.

    During each iteration return an array whose elements c obey:
    mean -std*lo < c < mean + std*hi

    where mean/std are the mean std of the input array.

    Parameters
    ----------
    arr : np.array
        A numpy array of numeric types.
    lo : float
        The negative clipping level.
    hi : float
        The positive clipping level.
    reps : int
        The number of iterations to perform. Default = 3.

    Returns
    -------
    mean : float
        The mean of the array, possibly nan
    std : float
        The std of the array, possibly nan

    Notes
    -----
    Scipy v0.16 now contains a comparable method that will ignore nan/inf
    values.
    """
    clipped = arr[np.isfinite(arr)]

    if len(clipped) < 1:
        return np.nan, np.nan

    std = np.std(clipped)
    mean = np.mean(clipped)
    prev_valid = len(clipped)
    for count in range(int(reps)):
        mask = (clipped > mean - std * lo) & (clipped < mean + std * hi)
        clipped = clipped[mask]

        curr_valid = len(clipped)
        if curr_valid < 1:
            break
        # No change in statistics if no change is noted
        if prev_valid == curr_valid:
            break
        std = np.std(clipped)
        mean = np.mean(clipped)
        prev_valid = curr_valid

    return mean, std


def _sf2(args):
    """
    A shallow wrapper for sigma_filter.

    Parameters
    ----------
    args : list
        A list of arguments for sigma_filter

    Returns
    -------
    None
    """
    # an easier to debug traceback when multiprocessing
    # thanks to https://stackoverflow.com/a/16618842/1710603
    try:
        return sigma_filter(*args)
    except Exception as e:
        import traceback

        logger.warning(e)
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def sigma_filter(filename, region, step_size, box_size, shape, domask, cube_index=None):
    """
    Calculate the background and rms for a sub region of an image. The results
    are written to shared memory - irms and ibkg.

    Parameters
    ----------
    filename : string
        Fits file to open

    region : list
        Region within the fits file that is to be processed. (row_min,
        row_max).

    step_size : (int, int)
        The filtering step size

    box_size : (int, int)
        The size of the box over which the filter is applied (each step).

    shape : (int, int, int)
        The shape of the fits cube (1,y,x) for an image.

    domask : bool
        If true then copy the data mask to the output.

    cube_index : int or None
        The index into the 3rd dimension (frequency?) to process.
        Default = None => process all.

    Returns
    -------
    None
    """

    ymin, ymax = region
    logger.debug(
        "rows {0}-{1} starting at {2}".format(
            ymin, ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())
        )
    )

    # cut out the region of interest plus 1/2 the box size
    # and clip to the image size
    data_row_min = max(0, ymin - box_size[0] // 2)
    data_row_max = min(shape[1], ymax + box_size[0] // 2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]

    sz = slice(None)
    sy = slice(data_row_min, data_row_max)
    sx = slice(None)

    if cube_index:
        sz = slice(cube_index, cube_index + 1)

    # For some reason we can't memmap a file with BSCALE not 1.0
    # so we ignore it now and scale it later
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as a:
        if NAXIS == 2:
            data = a[0].section[sy, sx]
            # ensure that we always end up with a 3d image
            data = data[None, :, :]
        elif NAXIS == 3:
            data = a[0].section[sz, sy, sx]
        elif NAXIS == 4:
            data = np.squeeze(a[0].section[0, sz, sy, sx])
        else:
            logger.error(f"Too many NAXIS for me {NAXIS}")
            logger.error("fix your file to be more sane")
            raise Exception("Too many NAXIS")

    # Manually scale the data if BSCALE is not 1.0
    header = fits.getheader(filename)
    if "BSCALE" in header:
        data *= header["BSCALE"]

    # force float64 for consistency
    data = data.astype(np.float64)

    logger.debug(f"data size is {data.shape}")
    logger.debug(f"data format is {data.dtype}")

    def box(r, c):
        """
        calculate the boundaries of the box centered at r,c
        with size = box_size
        """
        r_min = max(0, r - box_size[0] // 2)
        r_max = min(data.shape[1] - 1, r + box_size[0] // 2)
        c_min = max(0, c - box_size[1] // 2)
        c_max = min(data.shape[2] - 1, c + box_size[1] // 2)
        return r_min, r_max, c_min, c_max

    # set up a grid of rows/cols at which we will compute the bkg/rms
    rows = list(range(ymin - data_row_min, ymax - data_row_min, step_size[0]))
    rows.append(ymax - data_row_min)

    cols = list(range(0, shape[2], step_size[1]))
    cols.append(shape[2])

    # Find the shared memory and create a numpy array interface
    ibkg_shm = SharedMemory(name=f"ibkg_{memory_id}", create=False)
    ibkg = np.ndarray(shape, dtype=np.float64, buffer=ibkg_shm.buf)
    irms_shm = SharedMemory(name=f"irms_{memory_id}", create=False)
    irms = np.ndarray(shape, dtype=np.float64, buffer=irms_shm.buf)

    for k in range(shape[0]):
        logger.debug(f"Working on slice {k}")
        # store the computed bkg/rms in this smaller array
        vals = np.zeros(shape=(len(rows), len(cols)))

        # loop over
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                r_min, r_max, c_min, c_max = box(row, col)
                new = data[k, r_min:r_max, c_min:c_max]
                new = np.ravel(new)
                bkg, _ = sigmaclip(new, 3, 3)
                vals[i, j] = bkg

        # indices of all the pixels within our region
        gr, gc = np.mgrid[ymin - data_row_min : ymax - data_row_min, 0 : shape[2]]
        logger.debug(f"gr has shape {gr.shape}")
        logger.debug("Interpolating bkg to sharemem")
        ifunc = RegularGridInterpolator((rows, cols), vals)
        interp_bkg = np.array(ifunc((gr, gc)), dtype=np.float64)
        ibkg[k, ymin:ymax, :] = interp_bkg
        del ifunc, interp_bkg
        logger.debug(" ... done writing bkg")

        # wait for all to complete
        i = barrier.wait()
        if i == 0:
            barrier.reset()

        logger.debug("background subtraction")
        logger.debug(f"data shape {data.shape}, ibkg shape {ibkg.shape}")
        logger.debug(
            f"k {k}, ymin {ymin}, ymax {ymax}, data_row_min {data_row_min}, data_row_max {data_row_max}"
        )
        logger.debug(
            f"data slice is {0 + ymin - data_row_min}:{data.shape[1] - (data_row_max - ymax)}"
        )
        data[
            k, 0 + ymin - data_row_min : data.shape[1] - (data_row_max - ymax), :
        ] -= ibkg[k, ymin:ymax, :]
        logger.debug(".. done ")

        # reset/recycle the vals array
        vals[:] = 0

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                r_min, r_max, c_min, c_max = box(row, col)
                new = data[k, r_min:r_max, c_min:c_max]
                new = np.ravel(new)
                _, rms = sigmaclip(new, 3, 3)
                vals[i, j] = rms

        logger.debug("Interpolating rms to sharemem")
        ifunc = RegularGridInterpolator((rows, cols), vals)
        interp_rms = np.array(ifunc((gr, gc)), dtype=np.float64)
        irms[k, ymin:ymax, :] = interp_rms
        del ifunc, interp_rms
        logger.debug(" .. done writing rms")

        if domask:
            # wait for all to complete
            i = barrier.wait()
            if i == 0:
                barrier.reset()

            logger.debug("applying mask")
            mask = ~np.isfinite(
                data[
                    k,  # TODO: CHECK HERE
                    0 + ymin - data_row_min : data.shape[1] - (data_row_max - ymax),
                    :,
                ]
            )
            ibkg[k, ymin:ymax, :][mask] = np.nan
            irms[k, ymin:ymax, :][mask] = np.nan
            logger.debug("... done applying mask")

    logger.debug(
        "rows {0}-{1} finished at {2}".format(
            ymin, ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())
        )
    )
    return


def filter_mc_sharemem(
    filename,
    step_size,
    box_size,
    cores,
    shape,
    nslice=None,
    domask=True,
    cube_index=0,
    as_cube=False,
):
    """
    Calculate the background and noise images corresponding to the input file.
    The calculation is done via a box-car approach and uses multiple cores and
    shared memory.

    Parameters
    ----------
    filename : str
        Filename to be filtered.

    step_size : (int, int)
        Step size for the filter.

    box_size : (int, int)
        Box size for the filter.

    cores : int
        Number of cores to use. If None then use all available.

    shape : (int, int) or (int, int, int)
        The shape of the image or cube in the given file.

    nslice : int
        The image will be divided into this many horizontal stripes for
        processing. Default = None = equal to cores

    domask : bool
        True(Default) = copy data mask to output.

    cube_index : int
        For 3d data use this index into the third dimension. Default = 0

    as_cube : bool
        If true and the input data are 3d then compute bkg/rms over all channels.

    Returns
    -------
    bkg, rms : numpy.ndarray
        The interpolated background and noise images.
    """

    if cores is None:
        cores = multiprocessing.cpu_count()
    if (nslice is None) or (cores == 1):
        nslice = cores

    # force a 3d shape for consistency of processing
    if not as_cube:
        shape = (1, shape[0], shape[1])

    img_y = shape[1]

    logger.info("using {0} cores".format(cores))
    logger.info("using {0} stripes".format(nslice))

    if nslice > 1:
        # box widths should be multiples of the step_size, and not zero
        width_y = int(max(img_y / nslice / step_size[1], 1) * step_size[1])

        # locations of the box edges
        ymins = list(range(0, img_y, width_y))
        ymaxs = list(range(width_y, img_y, width_y))
        ymaxs.append(img_y)
    else:
        ymins = [0]
        ymaxs = [img_y]

    logger.debug("ymins {0}".format(ymins))
    logger.debug("ymaxs {0}".format(ymaxs))

    args = []
    for region in zip(ymins, ymaxs):
        args.append((filename, region, step_size, box_size, shape, domask, cube_index))

    exit = False
    try:
        global memory_id
        memory_id = str(uuid.uuid4())
        if (
            "Darwin" in platform.system()
        ):  # Some python installs on OSX limit filenames to 32 chars
            memory_id = memory_id[:23]
        nbytes = np.prod(shape) * np.float64(1).nbytes
        ibkg = SharedMemory(name=f"ibkg_{memory_id}", create=True, size=nbytes)
        irms = SharedMemory(name=f"irms_{memory_id}", create=True, size=nbytes)

        # start a new process for each task, hopefully to reduce residual
        # memory use
        method = "spawn"
        if sys.platform.startswith("linux"):
            method = "fork"
        ctx = multiprocessing.get_context(method)
        barrier = ctx.Barrier(parties=len(ymaxs))
        pool = ctx.Pool(
            processes=cores,
            maxtasksperchild=1,
            initializer=init,
            initargs=(barrier, memory_id),
        )
        try:
            # chunksize=1 ensures that we only send a single task to each
            # process
            pool.map_async(_sf2, args, chunksize=1).get(timeout=10000000)
        except KeyboardInterrupt:
            logger.error("Caught keyboard interrupt")
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            bkg = np.ndarray(shape, buffer=ibkg.buf, dtype=np.float64).astype(
                np.float32
            )
            rms = np.ndarray(shape, buffer=irms.buf, dtype=np.float64).astype(
                np.float32
            )
    finally:
        ibkg.close()
        ibkg.unlink()
        irms.close()
        irms.unlink()
        if exit:
            sys.exit(1)
    return bkg, rms


def filter_image(
    im_name,
    out_base,
    step_size=None,
    box_size=None,
    cores=None,
    mask=True,
    compressed=False,
    nslice=None,
    cube_index=None,
    as_cube=False,
):
    """
    Create a background and noise image from an input image. Resulting images
    are written to `outbase_bkg.fits` and `outbase_rms.fits`

    Parameters
    ----------
    im_name : str
        Image to filter.

    out_base : str or None
        The output filename base. Will be modified to make _bkg and _rms files.
        If None, then no files are written.

    step_size : (int,int)
        Tuple of the x,y step size in pixels

    box_size : (int,int)
        The size of the box in pixels

    cores : int
        Number of CPU corse to use. Default = all available

    nslice : int
        The image will be divided into this many horizontal stripes for
        processing. Default = None = equal to cores

    mask : bool
        Mask the output array to contain np.nna wherever the input array is nan
        or not finite. Default = true

    compressed : bool
        Return a compressed version of the background/noise images. Default =
        False

    cube_index : int
        If the input data is 3d, then use this index for the 3rd dimension.
        Default = None, use the first index.

    as_cube : bool
        If the input data is 3d, then compute a per channel bkg/rms.

    Returns
    -------
    bkg, rms : `numpy.ndarray`
        The computed background and rms maps (not compressed)
    """
    # Use the first slice of the 3rd dimension if not specified
    if cube_index is None:
        cube_index = 0

    header = fits.getheader(im_name)
    shape = (header["NAXIS2"], header["NAXIS1"])
    if "NAXIS3" in header:
        shape = (header["NAXIS3"], header["NAXIS2"], header["NAXIS1"])

    naxis = header["NAXIS"]
    if naxis > 2:
        naxis3 = header["NAXIS3"]
        if cube_index >= naxis3:
            logger.error(
                f"3rd dimension has len {naxis3} but index {cube_index} was passed"
            )
            return None

    if naxis == 2 and as_cube:
        logger.error(f"As_cube was set true but the data only have {naxis} axes")
        return None

    if step_size is None:
        step_size = get_step_size(header)

    if box_size is None:
        # default to 6x the step size so we have ~ 30beams
        box_size = (step_size[0] * 6, step_size[1] * 6)

    if compressed:
        if not step_size[0] == step_size[1]:
            step_size = (min(step_size), min(step_size))
            logger.info(
                f"Changing grid to be {step_size} so we can compress the output"
            )

    logger.info("using grid_size {0}, box_size {1}".format(step_size, box_size))
    logger.info("on data shape {0}".format(shape))
    bkg, rms = filter_mc_sharemem(
        im_name,
        step_size=step_size,
        box_size=box_size,
        cores=cores,
        shape=shape,
        nslice=nslice,
        domask=mask,
        cube_index=cube_index,
        as_cube=as_cube,
    )
    logger.info("done")

    if out_base is not None:
        # add a comment to the fits header
        header["HISTORY"] = "BANE {0}-({1})".format(__version__, __date__)

        bkg_out = "_".join([os.path.expanduser(out_base), "bkg.fits"])
        rms_out = "_".join([os.path.expanduser(out_base), "rms.fits"])

        # Test for BSCALE and scale back if needed before we write to a file
        bscale = 1.0
        if "BSCALE" in header:
            bscale = header["BSCALE"]

        # compress
        if compressed:
            hdu = fits.PrimaryHDU(bkg / bscale)
            hdu.header = copy.deepcopy(header)
            hdulist = fits.HDUList([hdu])
            compress(hdulist, step_size[0], bkg_out)
            hdulist[0].header = copy.deepcopy(header)
            hdulist[0].data = rms / bscale
            compress(hdulist, step_size[0], rms_out)
        else:
            write_fits(bkg / bscale, header, bkg_out)
            write_fits(rms / bscale, header, rms_out)

    return bkg, rms


###
# Helper functions
###
def get_step_size(header):
    """
    Determine the grid spacing for BANE operation.

    This is set to being 4x the synthesized beam width.
    If the beam is not circular then the "width" is sqrt(a*b)

    For the standard 4 pix/beam, the step size will be 16 pixels.

    Parameters
    ----------
    header

    Returns
    -------
    step_size : (int, int)
        The grid spacing for BANE operation
    """
    if "BMAJ" in header and "BMIN" in header:
        beam_size = np.sqrt(abs(header["BMAJ"] * header["BMIN"]))
        if "CDELT1" in header:
            pix_scale = np.sqrt(abs(header["CDELT1"] * header["CDELT2"]))
        elif "CD1_1" in header:
            pix_scale = np.sqrt(abs(header["CD1_1"] * header["CD2_2"]))
            if "CD1_2" in header and "CD2_1" in header:
                if header["CD1_2"] != 0 or header["CD2_1"] != 0:
                    logger.warning(
                        "CD1_2 and/or CD2_1 are non-zero and "
                        + "I don't know what to do with them"
                    )
                    logger.warning("Ingoring them")
        else:
            logger.warning("Cannot determine pixel scale, assuming 4 pixels per beam")
            pix_scale = beam_size / 4.0
        # default to 4x the synthesized beam width
        step_size = int(np.ceil(4 * beam_size / pix_scale))
    else:
        logger.info("BMAJ and/or BMIN not in fits header.")
        logger.info("Assuming 4 pix/beam, so we have step_size = 16 pixels")
        step_size = 16
    step_size = (step_size, step_size)
    return step_size


def write_fits(data, header, file_name):
    """
    Combine data and a fits header to write a fits file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be written.

    header : astropy.io.fits.hduheader
        The header for the fits file.

    file_name : string
        The file to write

    Returns
    -------
    None
    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_name, overwrite=True)
    logger.info("Wrote {0}".format(file_name))
    return
