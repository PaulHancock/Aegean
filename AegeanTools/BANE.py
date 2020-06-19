#! /usr/bin/env python

"""
This module contains all of the BANE specific code
The function filter_image should be imported from elsewhere and run as is.
"""
from __future__ import division
# standard imports
from astropy.io import fits
import copy
import logging
import multiprocessing
import multiprocessing.sharedctypes
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import sys
from tempfile import NamedTemporaryFile
from time import gmtime, strftime

# Aegean tools
from .fits_interp import compress

__author__ = 'Paul Hancock'
__version__ = 'v1.8.1'
__date__ = '2019-12-13'

# global variables for multiprocessing
ibkg = irms = None
bkg_events = []
mask_events = []


def barrier(events, sid, kind='neighbour'):
    """
    act as a multiprocessing barrier
    """
    events[sid].set()
    # only wait for the neighbours
    if kind=='neighbour':
        if sid > 0:
            logging.debug("{0} is waiting for {1}".format(sid, sid - 1))
            events[sid - 1].wait()
        if sid < len(bkg_events) - 1:
            logging.debug("{0} is waiting for {1}".format(sid, sid + 1))
            events[sid + 1].wait()
    # wait for all
    else:
        [e.wait() for e in events]
    return


def sigmaclip(arr, lo, hi, reps=3):
    """
    Perform sigma clipping on an array, ignoring non finite values.

    During each iteration return an array whose elements c obey:
    mean -std*lo < c < mean + std*hi

    where mean/std are the mean std of the input array.

    Parameters
    ----------
    arr : iterable
        An iterable array of numeric types.
    lo : float
        The negative clipping level.
    hi : float
        The positive clipping level.
    reps : int
        The number of iterations to perform.
        Default = 3.

    Returns
    -------
    mean : float
        The mean of the array, possibly nan
    std : float
        The std of the array, possibly nan

    Notes
    -----
    Scipy v0.16 now contains a comparable method that will ignore nan/inf values.
    """
    clipped = np.array(arr)[np.isfinite(arr)]

    if len(clipped) < 1:
        return np.nan, np.nan

    std = np.std(clipped)
    mean = np.mean(clipped)
    for _ in range(int(reps)):
        clipped = clipped[np.where(clipped > mean-std*lo)]
        clipped = clipped[np.where(clipped < mean+std*hi)]
        pstd = std
        if len(clipped) < 1:
            break
        std = np.std(clipped)
        mean = np.mean(clipped)
        if 2*abs(pstd-std)/(pstd+std) < 0.2:
            break
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
    except:
        import traceback
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def sigma_filter(filename, region, step_size, box_size, shape, domask, sid):
    """
    Calculate the background and rms for a sub region of an image. The results are
    written to shared memory - irms and ibkg.

    Parameters
    ----------
    filename : string
        Fits file to open

    region : list
        Region within the fits file that is to be processed. (row_min, row_max).

    step_size : (int, int)
        The filtering step size

    box_size : (int, int)
        The size of the box over which the filter is applied (each step).

    shape : tuple
        The shape of the fits image

    domask : bool
        If true then copy the data mask to the output.

    sid : int
        The stripe number

    Returns
    -------
    None
    """

    ymin, ymax = region
    logging.debug('rows {0}-{1} starting at {2}'.format(ymin, ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    # cut out the region of interest plus 1/2 the box size, but clip to the image size
    data_row_min = max(0, ymin - box_size[0]//2)
    data_row_max = min(shape[0], ymax + box_size[0]//2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]

    # For some reason we can't memmap a file with BSCALE not 1.0, so we signore it now and scale it later
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as a:
        if NAXIS == 2:
            data = a[0].section[data_row_min:data_row_max, 0:shape[1]]
        elif NAXIS == 3:
            data = a[0].section[0, data_row_min:data_row_max, 0:shape[1]]
        elif NAXIS == 4:
            data = a[0].section[0, 0, data_row_min:data_row_max, 0:shape[1]]
        else:
            logging.error("Too many NAXIS for me {0}".format(NAXIS))
            logging.error("fix your file to be more sane")
            raise Exception("Too many NAXIS")

    # Manually scale the data if BSCALE is not 1.0
    header = fits.getheader(filename)
    if 'BSCALE' in header:
        data *= header['BSCALE']

    row_len = shape[1]

    logging.debug('data size is {0}'.format(data.shape))

    def box(r, c):
        """
        calculate the boundaries of the box centered at r,c
        with size = box_size
        """
        r_min = max(0, r - box_size[0] // 2)
        r_max = min(data.shape[0] - 1, r + box_size[0] // 2)
        c_min = max(0, c - box_size[1] // 2)
        c_max = min(data.shape[1] - 1, c + box_size[1] // 2)
        return r_min, r_max, c_min, c_max

    # set up a grid of rows/cols at which we will compute the bkg/rms
    rows = list(range(ymin-data_row_min, ymax-data_row_min, step_size[0]))
    rows.append(ymax-data_row_min)
    cols = list(range(0, shape[1], step_size[1]))
    cols.append(shape[1])

    # store the computed bkg/rms in this smaller array
    vals = np.zeros(shape=(len(rows),len(cols)))

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            r_min, r_max, c_min, c_max = box(row, col)
            new = data[r_min:r_max, c_min:c_max]
            new = np.ravel(new)
            bkg, _ = sigmaclip(new, 3, 3)
            vals[i,j] = bkg

    # indices of all the pixels within our region
    gr, gc = np.mgrid[ymin-data_row_min:ymax-data_row_min, 0:shape[1]]

    logging.debug("Interpolating bkg to sharemem")
    ifunc = RegularGridInterpolator((rows, cols), vals)
    for i in range(gr.shape[0]):
        row = np.array(ifunc((gr[i], gc[i])), dtype=np.float32)
        start_idx = np.ravel_multi_index((ymin+i, 0), shape)
        end_idx = start_idx + row_len
        ibkg[start_idx:end_idx] = row  # np.ctypeslib.as_ctypes(row)
    del ifunc
    logging.debug(" ... done writing bkg")

    # signal that the bkg is done for this region, and wait for neighbours
    barrier(bkg_events, sid)

    logging.debug("{0} background subtraction".format(sid))
    for i in range(data_row_max - data_row_min):
        start_idx = np.ravel_multi_index((data_row_min + i, 0), shape)
        end_idx = start_idx + row_len
        data[i, :] = data[i, :] - ibkg[start_idx:end_idx]
    # reset/recycle the vals array
    vals[:] = 0

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            r_min, r_max, c_min, c_max = box(row, col)
            new = data[r_min:r_max, c_min:c_max]
            new = np.ravel(new)
            _ , rms = sigmaclip(new, 3, 3)
            vals[i,j] = rms

    logging.debug("Interpolating rm to sharemem rms")
    ifunc = RegularGridInterpolator((rows, cols), vals)
    for i in range(gr.shape[0]):
        row = np.array(ifunc((gr[i], gc[i])), dtype=np.float32)
        start_idx = np.ravel_multi_index((ymin+i, 0), shape)
        end_idx = start_idx + row_len
        irms[start_idx:end_idx] = row  # np.ctypeslib.as_ctypes(row)
    del ifunc
    logging.debug(" .. done writing rms")

    if domask:
        barrier(mask_events, sid)
        logging.debug("applying mask")
        for i in range(gr.shape[0]):
            mask = np.where(np.bitwise_not(np.isfinite(data[i + ymin-data_row_min,:])))[0]
            for j in mask:
                idx = np.ravel_multi_index((i + ymin,j),shape)
                ibkg[idx] = np.nan
                irms[idx] = np.nan
        logging.debug(" ... done applying mask")
    logging.debug('rows {0}-{1} finished at {2}'.format(ymin, ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    return


def filter_mc_sharemem(filename, step_size, box_size, cores, shape, nslice=None, domask=True):
    """
    Calculate the background and noise images corresponding to the input file.
    The calculation is done via a box-car approach and uses multiple cores and shared memory.

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

    nslice : int
        The image will be divided into this many horizontal stripes for processing.
        Default = None = equal to cores

    shape : (int, int)
        The shape of the image in the given file.

    domask : bool
        True(Default) = copy data mask to output.

    Returns
    -------
    bkg, rms : numpy.ndarray
        The interpolated background and noise images.
    """

    if cores is None:
        cores = multiprocessing.cpu_count()
    if (nslice is None) or (cores==1):
        nslice = cores

    img_y, img_x = shape
    # initialise some shared memory
    global ibkg
    # bkg = np.ctypeslib.as_ctypes(np.empty(shape, dtype=np.float32))
    # ibkg = multiprocessing.sharedctypes.Array(bkg._type_, bkg, lock=True)
    ibkg = multiprocessing.Array('f', img_y*img_x)

    global irms
    #rms = np.ctypeslib.as_ctypes(np.empty(shape, dtype=np.float32))
    #irms = multiprocessing.sharedctypes.Array(rms._type_, rms, lock=True)
    irms = multiprocessing.Array('f', img_y * img_x)

    logging.info("using {0} cores".format(cores))
    logging.info("using {0} stripes".format(nslice))

    if nslice > 1:
        # box widths should be multiples of the step_size, and not zero
        width_y = int(max(img_y/nslice/step_size[1], 1) * step_size[1])

        # locations of the box edges
        ymins = list(range(0, img_y, width_y))
        ymaxs = list(range(width_y, img_y, width_y))
        ymaxs.append(img_y)
    else:
        ymins = [0]
        ymaxs = [img_y]

    logging.debug("ymins {0}".format(ymins))
    logging.debug("ymaxs {0}".format(ymaxs))

    # create an event per stripe
    global bkg_events, mask_events
    bkg_events = [multiprocessing.Event() for _ in range(len(ymaxs))]
    mask_events = [multiprocessing.Event() for _ in range(len(ymaxs))]

    args = []
    for i, region in enumerate(zip(ymins, ymaxs)):
        args.append((filename, region, step_size, box_size, shape, domask, i))

    # start a new process for each task, hopefully to reduce residual memory use
    pool = multiprocessing.Pool(processes=cores, maxtasksperchild=1)
    try:
        # chunksize=1 ensures that we only send a single task to each process
        pool.map_async(_sf2, args, chunksize=1).get(timeout=10000000)
    except KeyboardInterrupt:
        logging.error("Caught keyboard interrupt")
        pool.close()
        sys.exit(1)
    pool.close()
    pool.join()
    bkg = np.reshape(np.array(ibkg[:], dtype=np.float32), shape)
    rms = np.reshape(np.array(irms[:], dtype=np.float32), shape)
    del ibkg, irms
    return bkg, rms


def filter_image(im_name, out_base, step_size=None, box_size=None, twopass=False, cores=None, mask=True, compressed=False, nslice=None):
    """
    Create a background and noise image from an input image.
    Resulting images are written to `outbase_bkg.fits` and `outbase_rms.fits`

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

    twopass : bool
        Perform a second pass calculation to ensure that the noise is not contaminated by the background.
        Default = False

    cores : int
        Number of CPU corse to use.
        Default = all available

    nslice : int
        The image will be divided into this many horizontal stripes for processing.
        Default = None = equal to cores

    mask : bool
        Mask the output array to contain np.nna wherever the input array is nan or not finite.
        Default = true

    compressed : bool
        Return a compressed version of the background/noise images.
        Default = False

    Returns
    -------
    bkg, rms : `numpy.ndarray`
        The computed background and rms maps (not compressed)
    """

    header = fits.getheader(im_name)
    shape = (header['NAXIS2'],header['NAXIS1'])

    if step_size is None:
        step_size = get_step_size(header)

    if box_size is None:
        # default to 6x the step size so we have ~ 30beams
        box_size = (step_size[0]*6, step_size[1]*6)

    if compressed:
        if not step_size[0] == step_size[1]:
            step_size = (min(step_size), min(step_size))
            logging.info("Changing grid to be {0} so we can compress the output".format(step_size))

    logging.info("using grid_size {0}, box_size {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(shape))
    bkg, rms = filter_mc_sharemem(im_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, nslice=nslice, domask=mask)
    logging.info("done")

    if out_base is not None:
        # add a comment to the fits header
        header['HISTORY'] = 'BANE {0}-({1})'.format(__version__, __date__)

        bkg_out = '_'.join([os.path.expanduser(out_base), 'bkg.fits'])
        rms_out = '_'.join([os.path.expanduser(out_base), 'rms.fits'])

        # Test for BSCALE and scale back if needed before we write to a file
        bscale = 1.0
        if 'BSCALE' in header:
            bscale = header['BSCALE']

        # compress
        if compressed:
            hdu = fits.PrimaryHDU(bkg/bscale)
            hdu.header = copy.deepcopy(header)
            hdulist = fits.HDUList([hdu])
            compress(hdulist, step_size[0], bkg_out)
            hdulist[0].header = copy.deepcopy(header)
            hdulist[0].data = rms/bscale
            compress(hdulist, step_size[0], rms_out)
        else:
            write_fits(bkg/bscale, header, bkg_out)
            write_fits(rms/bscale, header, rms_out)

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
    if 'BMAJ' in header and 'BMIN' in header:
        beam_size = np.sqrt(abs(header['BMAJ']*header['BMIN']))
        if 'CDELT1' in header:
            pix_scale = np.sqrt(abs(header['CDELT1']*header['CDELT2']))
        elif 'CD1_1' in header:
            pix_scale = np.sqrt(abs(header['CD1_1']*header['CD2_2']))
            if 'CD1_2' in header and 'CD2_1' in header:
                if header['CD1_2'] != 0 or header['CD2_1']!=0:
                    logging.warning("CD1_2 and/or CD2_1 are non-zero and I don't know what to do with them")
                    logging.warning("Ingoring them")
        else:
            logging.warning("Cannot determine pixel scale, assuming 4 pixels per beam")
            pix_scale = beam_size/4.
        # default to 4x the synthesized beam width
        step_size = int(np.ceil(4*beam_size/pix_scale))
    else:
        logging.info("BMAJ and/or BMIN not in fits header.")
        logging.info("Assuming 4 pix/beam, so we have step_size = 16 pixels")
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
    logging.info("Wrote {0}".format(file_name))
    return
