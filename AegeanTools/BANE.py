#! /usr/bin/env python

"""
This module contains all of the BANE specific code
The function filter_image should be imported from elsewhere and run as is.
"""

# standard imports
from astropy.io import fits
import copy
import logging
import multiprocessing
import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator
import sys
from tempfile import NamedTemporaryFile
from time import gmtime, strftime

# Aegean tools
from .fits_interp import compress

__author__ = 'Paul Hancock'
__version__ = 'v1.5.0'
__date__ = '2018-05-05'

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
    return sigma_filter(*args)


def sigma_filter(filename, region, step_size, box_size, shape, dobkg=True):
    """
    Calculate the background and rms for a sub region of an image. The results are
    written to shared memory - irms and ibkg.

    Parameters
    ----------
    filename : string
        Fits file to open

    region : (float, float, float, float)
        Region within the fits file that is to be processed. (ymin, ymax, xmin, xmax).

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

    # Caveat emptor: The code that follows is very difficult to read.
    # xmax is not x_max, and x,y actually should be y,x
    # TODO: fix the code below so that the above comment can be removed

    ymin, ymax, xmin, xmax = region
    logging.debug('{0}x{1},{2}x{3} starting at {4}'.format(xmin, xmax, ymin, ymax,
                                                           strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    cmin = max(0, ymin - box_size[1]//2)
    cmax = min(shape[1], ymax + box_size[1]//2)
    rmin = max(0, xmin - box_size[0]//2)
    rmax = min(shape[0], xmax + box_size[0]//2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]
    # It seems that I cannot memmap the same file multiple times without errors
    with fits.open(filename, memmap=False) as a:
        if NAXIS == 2:
            data = a[0].section[rmin:rmax, cmin:cmax]
        elif NAXIS == 3:
            data = a[0].section[0, rmin:rmax, cmin:cmax]
        elif NAXIS == 4:
            data = a[0].section[0, 0, rmin:rmax, cmin:cmax]
        else:
            logging.error("Too many NAXIS for me {0}".format(NAXIS))
            logging.error("fix your file to be more sane")
            raise Exception("Too many NAXIS")

    # x/y min/max should refer to indices into data
    # this is the region over which we want to operate
    ymin -= cmin
    ymax -= cmin
    xmin -= rmin
    xmax -= rmin

    def locations(step_size, xmin, xmax, ymin, ymax):
        """
        Generator function to iterate over a grid of x,y coords
        operates only within the given bounds
        Returns:
        x, y
        """

        xvals = list(range(xmin, xmax, step_size[0]))
        if xvals[-1] != xmax:
            xvals.append(xmax)
        yvals = list(range(ymin, ymax, step_size[1]))
        if yvals[-1] != ymax:
            yvals.append(ymax)
        # initial data
        for y in yvals:
            for x in xvals:
                yield x, y

    def box(x, y):
        """
        calculate the boundaries of the box centered at x,y
        with size = box_size
        """
        x_min = int(max(0, x-box_size[0]/2))
        x_max = int(min(data.shape[0]-1, x+box_size[0]/2))
        y_min = int(max(0, y-box_size[1]/2))
        y_max = int(min(data.shape[1]-1, y+box_size[1]/2))
        return x_min, x_max, y_min, y_max

    bkg_points = []
    bkg_values = []
    rms_points = []
    rms_values = []

    for x, y in locations(step_size, xmin, xmax, ymin, ymax):
        x_min, x_max, y_min, y_max = box(x, y)
        new = data[x_min:x_max, y_min:y_max]
        new = np.ravel(new)
        new = sigmaclip(new, 3, 3)
        # If we are left with (or started with) no data, then just move on
        if len(new) < 1:
            continue

        if dobkg:
            bkg = np.median(new)
            bkg_points.append((x+rmin, y+cmin))  # these coords need to be indices into the larger array
            bkg_values.append(bkg)
        rms = np.std(new)
        rms_points.append((x+rmin, y+cmin))
        rms_values.append(rms)

    ymin, ymax, xmin, xmax = region
    gx, gy = np.mgrid[xmin:xmax, ymin:ymax]
    # If the bkg/rms calculation above didn't yield any points, then our interpolated values are all nans
    if len(rms_points) > 1:
        logging.debug("Interpolating rms")
        ifunc = LinearNDInterpolator(rms_points, rms_values)
        # force 32 bit floats
        interpolated_rms = np.array(ifunc((gx, gy)), dtype=np.float32)
        del ifunc
    else:
        interpolated_rms = np.empty((len(gx), len(gy)), dtype=np.float32)*np.nan
    with irms.get_lock():
        logging.debug("Writing rms to sharemem")
        for i, row in enumerate(interpolated_rms):
            start_idx = np.ravel_multi_index((xmin + i, ymin), shape)
            end_idx = start_idx + len(row)
            irms[start_idx:end_idx] = row
    logging.debug(" .. done writing rms")

    if dobkg:
        if len(bkg_points)>1:
            logging.debug("Interpolating bkg")
            ifunc = LinearNDInterpolator(bkg_points, bkg_values)
            interpolated_bkg = np.array(ifunc((gx, gy)), dtype=np.float32)
            del ifunc
        else:
            interpolated_bkg = np.empty((len(gx), len(gy)), dtype=np.float32)*np.nan
        with ibkg.get_lock():
            logging.debug("Writing bkg to sharemem")
            for i, row in enumerate(interpolated_bkg):
                start_idx = np.ravel_multi_index((xmin + i, ymin), shape)
                end_idx = start_idx + len(row)
                ibkg[start_idx:end_idx] = row
        logging.debug(" .. done writing bkg")
    logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin, xmax, ymin, ymax,
                                                           strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    return


def gen_factors(m, permute=True):
    """
    Generate a list of integer factors of the the input m.

    Parameters
    ----------
    m : int
        The number to factorise

    permute : bool
        If true then yield both x,y and y,x if x*y=m. Otherwise only yield one of the two. Default = True.

    Yields
    ------
    x, y : int
        Two integers x and y such that x*y=m
    """
    # convert to int if people have been naughty
    n = int(abs(m))
    # brute force the factors, one of which is always less than sqrt(n)
    for i in range(1, int(n**0.5+1)):
        if n % i == 0:
            yield i, n/i
            # yield the reverse pair if it is unique
            if i != n/i and permute:
                yield n/i, i


def optimum_sections(cores, data_shape):
    """
    Choose the best sectioning scheme based on the number of corse available and the shape of the data.
    "Best" here means minimum perimeter.

    Parameters
    ----------
    cores : int
        The number of corse which are going to be doing the processing.
    data_shape : tuple
        Shape of the data as (x,y).

    Returns
    -------
    nx, ny : int
        The number of divisions in each dimension.
    """
    if cores == 1:
        return (1, 1)
    if cores % 1 == 1:
        cores -= 1
    x, y = data_shape
    min_overlap = np.inf
    best = (1, 1)
    for (mx, my) in gen_factors(cores):
        overlap = x*(my-1) + y*(mx-1)
        if overlap < min_overlap:
            best = (mx, my)
            min_overlap = overlap
    logging.debug("Sectioning chosen to be {0[0]}x{0[1]} for a score of {1}".format(best, min_overlap))
    return best


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


def filter_mc_sharemem(filename, step_size, box_size, cores, shape, dobkg=True):
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

    img_y, img_x = shape
    # initialise some shared memory
    alen = shape[0]*shape[1]
    if dobkg:
        global ibkg
        ibkg = multiprocessing.Array('f', alen)
    else:
        ibkg = None
    global irms
    irms = multiprocessing.Array('f', alen)

    logging.info("using {0} cores".format(cores))
    nx, ny = optimum_sections(cores, shape)

    # box widths should be multiples of the step_size, and not zero
    width_x = int(max(img_x/nx/step_size[0], 1) * step_size[0])
    width_y = int(max(img_y/ny/step_size[1], 1) * step_size[1])

    xstart = width_x
    ystart = width_y
    xend = img_x - img_x % width_x  # the end point of the last "full" box
    yend = img_y - img_y % width_y

    # locations of the box edges
    xmins = [0]
    xmins.extend(list(range(xstart, xend, width_x)))

    xmaxs = [xstart]
    xmaxs.extend(list(range(xstart+width_x, xend+1, width_x)))
    xmaxs[-1] = img_x

    ymins = [0]
    ymins.extend(list(range(ystart, yend, width_y)))

    ymaxs = [ystart]
    ymaxs.extend(list(range(ystart+width_y, yend+1, width_y)))
    ymaxs[-1] = img_y

    args = []
    for xmin, xmax in zip(xmins, xmaxs):
        for ymin, ymax in zip(ymins, ymaxs):
            region = [xmin, xmax, ymin, ymax]
            args.append((filename, region, step_size, box_size, shape, dobkg))

    pool = multiprocessing.Pool(processes=cores)
    try:
        pool.map_async(_sf2, args).get(timeout=10000000)
    except KeyboardInterrupt:
        logging.error("Caught keyboard interrupt")
        pool.close()
        sys.exit(1)
    pool.close()
    pool.join()

    # reshape our 1d arrays back into a 2d image
    if dobkg:
        logging.debug("reshaping bkg")
        interpolated_bkg = np.reshape(np.array(ibkg[:], dtype=np.float32), shape)
        logging.debug(" bkg is {0}".format(interpolated_bkg.dtype))
        logging.debug(" ... done at {0}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    else:
        interpolated_bkg = None
    del ibkg
    logging.debug("reshaping rms")
    interpolated_rms = np.reshape(np.array(irms[:], dtype=np.float32), shape)
    logging.debug(" ... done at {0}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    del irms

    return interpolated_bkg, interpolated_rms


def filter_image(im_name, out_base, step_size=None, box_size=None, twopass=False, cores=None, mask=True, compressed=False):
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
    bkg, rms = filter_mc_sharemem(im_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape)
    logging.info("done")

    # force float 32s to avoid bloated files
    bkg = np.array(bkg, dtype=np.float32)
    rms = np.array(rms, dtype=np.float32)

    if twopass:
        # TODO: check what this does for our memory usage
        # Answer: The interpolation step peaks at about 5x the normal value.
        tempfile = NamedTemporaryFile(delete=False)
        data = fits.getdata(im_name) - bkg
        header = fits.getheader(im_name)
        # write 32bit floats to reduce memory overhead
        write_fits(np.array(data, dtype=np.float32), header, tempfile)
        tempfile.close()
        temp_name = tempfile.name
        del data, header, tempfile, rms
        logging.info("running second pass to get a better rms")
        junk, rms = filter_mc_sharemem(temp_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, dobkg=False)
        del junk
        rms = np.array(rms, dtype=np.float32)
        os.remove(temp_name)

    bkg_out = '_'.join([os.path.expanduser(out_base), 'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base), 'rms.fits'])


    # load the file since we are now going to fiddle with it
    header = fits.getheader(im_name)
    header['HISTORY'] = 'BANE {0}-({1})'.format(__version__, __date__)
    if compressed:
        hdu = fits.PrimaryHDU(bkg)
        hdu.header = copy.deepcopy(header)
        hdulist = fits.HDUList([hdu])
        compress(hdulist, step_size[0], bkg_out)
        hdulist[0].header = copy.deepcopy(header)
        hdulist[0].data = rms
        compress(hdulist, step_size[0], rms_out)
        return
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
