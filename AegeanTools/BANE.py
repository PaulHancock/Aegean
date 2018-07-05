#! /usr/bin/env python

"""
This module contains all of the BANE specific code
The function filter_image should be imported from elsewhere and run as is.
"""

__author__ = 'Paul Hancock'
__version__ = 'v1.6.5'
__date__ = '2018-07-05'

# standard imports
from astropy.io import fits
import copy
import logging
import multiprocessing
import multiprocessing.sharedctypes
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator
import sys
from tempfile import NamedTemporaryFile
from time import gmtime, strftime

# Aegean tools
from .fits_interp import compress

# global variables for multiprocessing
ibkg = irms = None

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
    clipped : numpy.array
        The clipped array.
        The clipped array may be empty!

    Notes
    -----
    Scipy v0.16 now contains a comparable method that will ignore nan/inf values.
    """
    clipped = np.array(arr)[np.isfinite(arr)]

    if len(clipped) < 1:
        return clipped

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
    return clipped


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


def sigma_filter(filename, region, step_size, box_size, shape, dobkg=True):
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

    dobkg : bool
        Do a background calculation. If false then only the rms is calculated. Default = True.

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

    with fits.open(filename, memmap=True) as a:
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

    logging.debug('data size is {0}'.format(data.shape))

    def locations(step, _r_min, _r_max, _c_min, _c_max):
        """
        Generator function to iterate over a grid of r,c coords
        operates only within the given bounds
        Returns:
        r, c
        """

        rvals = list(range(_r_min, _r_max, step[0]))
        if rvals[-1] != _r_max:
            rvals.append(_r_max)
        cvals = list(range(_c_min, _c_max, step[1]))
        if cvals[-1] != _c_max:
            cvals.append(_c_max)
        # initial data
        for c in cvals:
            for r in rvals:
                yield r, c

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

    # lists that hold the calculated values and array coordinates
    bkg_points = []
    bkg_values = []
    rms_points = []
    rms_values = []

    for row, col in locations(step_size, ymin-data_row_min, ymax-data_row_min, 0, shape[1]):
        r_min, r_max, c_min, c_max = box(row, col)
        new = data[r_min:r_max, c_min:c_max]
        new = np.ravel(new)
        new = sigmaclip(new, 3, 3)
        # If we are left with (or started with) no data, then just move on
        if len(new) < 1:
            continue

        if dobkg:
            bkg = np.median(new)
            bkg_points.append((row + data_row_min, col))  # these coords need to be indices into the larger array
            bkg_values.append(bkg)
        rms = np.std(new)
        rms_points.append((row + data_row_min, col))
        rms_values.append(rms)

    # indices of the shape we want to write to (not the shape of data)
    gx, gy = np.mgrid[ymin:ymax, 0:shape[1]]
    # If the bkg/rms calculation above didn't yield any points, then our interpolated values are all nans
    if len(rms_points) > 1:
        logging.debug("Interpolating rms")
        ifunc = LinearNDInterpolator(rms_points, rms_values)
        # force 32 bit floats
        interpolated_rms = np.array(ifunc((gx, gy)), dtype=np.float32)
        del ifunc
    else:
        logging.debug("rms is all nans")
        interpolated_rms = np.empty(gx.shape, dtype=np.float32)*np.nan

    with irms.get_lock():
        logging.debug("Writing rms to sharemem")
        for i, row in enumerate(interpolated_rms):
            irms[i + ymin] = np.ctypeslib.as_ctypes(row)
    logging.debug(" .. done writing rms")

    if dobkg:
        if len(bkg_points)>1:
            logging.debug("Interpolating bkg")
            ifunc = LinearNDInterpolator(bkg_points, bkg_values)
            interpolated_bkg = np.array(ifunc((gx, gy)), dtype=np.float32)
            del ifunc
        else:
            logging.debug("bkg is all nans")
            interpolated_bkg = np.empty(gx.shape, dtype=np.float32)*np.nan
        with ibkg.get_lock():
            logging.debug("Writing bkg to sharemem")
            for i, row in enumerate(interpolated_bkg):
                ibkg[i + ymin] = np.ctypeslib.as_ctypes(row)
        logging.debug(" .. done writing bkg")
    logging.debug('rows {0}-{1} finished at {2}'.format(ymin, ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    del bkg_points, bkg_values
    del rms_points, rms_values

    return


def mask_img(data, mask_data):
    """
    Take two images of the same shape, and transfer the mask from one to the other.
    Masking is done via any not finite values. The mask value is numpy.nan.

    Parameters
    ----------
    data : numpy.ndarray
        A 2d array of data that is to be masked. This array is modified in place.
    mask_data : numpy.ndarray
        A 2d array of data that contains some not finite value which are to be used to mask the input data.

    Returns
    -------
    None
    """
    mask = np.where(np.isnan(mask_data))
    # If the input image has more than 2 dimensions then the mask has too many dimensions
    # our data has only 2d so we use just the last two dimensions of the mask.
    if len(mask) > 2:
        mask = mask[-2], mask[-1]
        logging.debug("mask = {0}".format(mask))
    try:
        data[mask] = np.NaN
    except IndexError:
        logging.info("failed to mask file, not a critical failure")


def filter_mc_sharemem(filename, step_size, box_size, cores, shape, dobkg=True, nslice=8):
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

    dobkg : bool
        If True then calculate the background, otherwise assume it is zero.

    Returns
    -------
    bkg, rms : numpy.ndarray
        The interpolated background and noise images.
    """

    if cores is None:
        cores = multiprocessing.cpu_count()
    if nslice is None:
        nslice = cores

    img_y, img_x = shape
    # initialise some shared memory
    if dobkg:
        global ibkg
        bkg = np.ctypeslib.as_ctypes(np.empty(shape, dtype=np.float32))
        ibkg = multiprocessing.sharedctypes.Array(bkg._type_, bkg, lock=True)
    else:
        bkg = None
        ibkg = None
    global irms
    rms = np.ctypeslib.as_ctypes(np.empty(shape, dtype=np.float32))
    irms = multiprocessing.sharedctypes.Array(rms._type_, rms, lock=True)

    logging.info("using {0} cores".format(cores))
    logging.info("using {0} stripes".format(nslice))
    # Use a striped sectioning scheme
    ny = nslice

    # box widths should be multiples of the step_size, and not zero
    width_y = int(max(img_y/ny/step_size[1], 1) * step_size[1])

    ystart = width_y
    yend = img_y - img_y % width_y

    # locations of the box edges
    ymins = [0]
    ymins.extend(list(range(ystart, yend, width_y)))

    ymaxs = [ystart]
    ymaxs.extend(list(range(ystart+width_y, yend+1, width_y)))
    ymaxs[-1] = img_y

    logging.debug("ymins {0}".format(ymins))
    logging.debug("ymaxs {0}".format(ymaxs))

    args = []
    for ymin, ymax in zip(ymins, ymaxs):
        region = (ymin, ymax)
        args.append((filename, region, step_size, box_size, shape, dobkg))

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

    rms = np.array(irms)
    if dobkg:
        bkg = np.array(ibkg)
    return bkg, rms


def filter_image(im_name, out_base, step_size=None, box_size=None, twopass=False, cores=None, mask=True, compressed=False, nslice=None):
    """
    Create a background and noise image from an input image.
    Resulting images are written to `outbase_bkg.fits` and `outbase_rms.fits`

    Parameters
    ----------
    im_name : str or HDUList
        Image to filter. Either a string filename or an astropy.io.fits.HDUList.
    out_base : str
        The output filename base. Will be modified to make _bkg and _rms files.
    step_size : (int,int)
        Tuple of the x,y step size in pixels
    box_size : (int,int)
        The size of the box in piexls
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
    None

    """

    header = fits.getheader(im_name)
    shape = (header['NAXIS2'],header['NAXIS1'])

    if step_size is None:
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

    if box_size is None:
        # default to 6x the step size so we have ~ 30beams
        box_size = (step_size[0]*6, step_size[1]*6)

    if compressed:
        if not step_size[0] == step_size[1]:
            step_size = (min(step_size), min(step_size))
            logging.info("Changing grid to be {0} so we can compress the output".format(step_size))

    logging.info("using grid_size {0}, box_size {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(shape))
    bkg, rms = filter_mc_sharemem(im_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, nslice=nslice)
    logging.info("done")

    # force float 32s to avoid bloated files
    bkg = np.array(bkg, dtype=np.float32)
    rms = np.array(rms, dtype=np.float32)

    if twopass:
        # TODO: check what this does for our memory usage
        # Answer: The interpolation step peaks at about 5x the normal value.
        tempfile = NamedTemporaryFile(delete=False)
        data = fits.getdata(im_name) - bkg
        # write 32bit floats to reduce memory overhead
        write_fits(np.array(data, dtype=np.float32), header, tempfile)
        tempfile.close()
        temp_name = tempfile.name
        del data, tempfile, rms
        logging.info("running second pass to get a better rms")
        junk, rms = filter_mc_sharemem(temp_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, dobkg=False, nslice=nslice)
        del junk
        rms = np.array(rms, dtype=np.float32)
        os.remove(temp_name)

    bkg_out = '_'.join([os.path.expanduser(out_base), 'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base), 'rms.fits'])


    # add a comment to the fits header
    header['HISTORY'] = 'BANE {0}-({1})'.format(__version__, __date__)

    # compress
    if compressed:
        hdu = fits.PrimaryHDU(bkg)
        hdu.header = copy.deepcopy(header)
        hdulist = fits.HDUList([hdu])
        compress(hdulist, step_size[0], bkg_out)
        hdulist[0].header = copy.deepcopy(header)
        hdulist[0].data = rms
        compress(hdulist, step_size[0], rms_out)
        return

    # mask
    if mask:
        ref = fits.getdata(im_name)
        mask_img(bkg, ref)
        mask_img(rms, ref)
        del ref
    write_fits(bkg, header, bkg_out)
    write_fits(rms, header, rms_out)


###
# Helper functions
###
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
