#! /usr/bin/env python

# standard imports
import numpy as np
import sys
import os
from optparse import OptionParser
import time
from time import gmtime, strftime
import logging
import copy
from tempfile import NamedTemporaryFile

from scipy.interpolate import LinearNDInterpolator
from astropy.io import fits


# Aegean tools
import AegeanTools.pprocess as pprocess
from AegeanTools.fits_interp import compress

import multiprocessing

__author__ = 'Paul Hancock'
__version__ = 'v1.3'
__date__ = '2015-10-12'


def sigmaclip(arr, lo, hi, reps = 3):
    """
    Perform sigma clipping on an array.
    Return an array whose elements c obey:
     mean - std*lo < c < mean + std*hi
    where mean/std refers to the mean/std of the input array.

    I'd like scipy to do this, but it appears that only scipy v0.16+ has a useful sigmaclip function.

    :param arr: Input array
    :param lo: Lower limit (mean -std*lo)
    :param hi: Upper limit (mean +std*hi)
    :param reps: maximum number of repetitions of the clipping
    :return: clipped array
    """
    clipped = arr[np.isfinite(arr)]
    std = np.std(clipped)
    mean = np.mean(clipped)
    for i in xrange(reps):
        clipped = clipped[np.where(clipped > mean-std*lo)]
        clipped = clipped[np.where(clipped < mean+std*hi)]
        pstd = std
        std = np.std(clipped)
        mean = np.mean(clipped)
        if 2*abs(pstd-std)/(pstd+std) < 0.2:
            break
    return clipped


def sigma_filter(filename, region, step_size, box_size, shape, ibkg=None, irms=None):
    """

    :param filename:
    :param region:
    :param step_size:
    :param box_size:
    :param shape:
    :param ibkg:
    :param irms:
    :return:
    """

    # Caveat emptor: The code that follows is very difficult to read.
    # xmax is not x_max, and x,y actually should be y,x
    # TODO: fix the code below so that the above comment can be removed

    ymin, ymax, xmin, xmax = region

    logging.debug('{0}x{1},{2}x{3} starting at {4}'.format(xmin, xmax, ymin, ymax,
                                                           strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    cmin = max(0, ymin - box_size[1]/2)
    cmax = min(shape[1], ymax + box_size[1]/2)
    rmin = max(0, xmin - box_size[0]/2)
    rmax = min(shape[0], xmax + box_size[0]/2)

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
            sys.exit(1)

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

        xvals = range(xmin, xmax, step_size[0])
        if xvals[-1] != xmax:
            xvals.append(xmax)
        yvals = range(ymin, ymax, step_size[1])
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
        x_min = max(0, x-box_size[0]/2)
        x_max = min(data.shape[0]-1, x+box_size[0]/2)
        y_min = max(0, y-box_size[1]/2)
        y_max = min(data.shape[1]-1, y+box_size[1]/2)
        return x_min, x_max, y_min, y_max

    bkg_points = []
    rms_points = []
    bkg_values = []
    rms_values = []

    for x, y in locations(step_size, xmin, xmax, ymin, ymax):
        x_min, x_max, y_min, y_max = box(x, y)
        new = data[x_min:x_max, y_min:y_max]
        new = np.ravel(new)
        new = sigmaclip(new, 3, 3)
        bkg = np.median(new)
        rms = np.std(new)

        if bkg is not None:
            bkg_points.append((x+rmin, y+cmin))  # these coords need to be indices into the larger array
            bkg_values.append(bkg)
        if rms is not None:
            rms_points.append((x+rmin, y+cmin))
            rms_values.append(rms)

    ymin, ymax, xmin, xmax = region
    # check if we have been passed some shared memory references
    # and if so do the interpolation
    # otherwise pass back our coords and lists so that interpolation can be done elsewhere
    if ibkg is not None and irms is not None:
        gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
        ifunc = LinearNDInterpolator(rms_points, rms_values)
        interpolated_rms = ifunc((gx, gy))
        with irms.get_lock():
            for i,row in enumerate(interpolated_rms):
                start_idx = np.ravel_multi_index((xmin + i, ymin), shape)
                end_idx = start_idx + len(row)
                irms[start_idx:end_idx] = row

        ifunc = LinearNDInterpolator(bkg_points, bkg_values)
        interpolated_bkg = ifunc((gx, gy))
        with ibkg.get_lock():
            for i,row in enumerate(interpolated_bkg):
                start_idx = np.ravel_multi_index((xmin + i,ymin), shape)
                end_idx = start_idx + len(row)
                ibkg[start_idx:end_idx] = row
        logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin, xmax, ymin, ymax,
                                                               strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        return
    else:
        logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin, xmax, ymin, ymax,
                                                               strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        return xmin, xmax, ymin, ymax, bkg_points, bkg_values, rms_points, rms_values


def running_filter(filename, region, step_size, box_size, shape, ibkg=None, irms=None):
    """
    Perform a running filter over a region within a file.
    The region can be a sub set of the data within the file - only the useful data will be loaded.

    :param filename: File from which to extract data
    :param region: [xmin,xmax,ymin,ymax] indices over which we are to operate
    :param step_size: amount to move filtering box each iteration
    :param box_size: Size of filtering box
    :return: xmin, xmax, ymin, ymax, bkg_points, bkg_values, rms_points, rms_values
    """
    # Avoid importing this code unless we have been instructed to use the RP method.
    # this means that we are not longer reliant on the blist module
    from AegeanTools.running_percentile import RunningPercentiles as RP

    # Caveat emptor: The code that follows is very difficult to read.
    # xmax is not x_max, and x,y actually should be y,x
    # TODO: fix the code below so that the above comment can be removed

    ymin,ymax,xmin,xmax = region

    logging.debug('{0}x{1},{2}x{3} starting at {4}'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    cmin = max(0, ymin - box_size[1]/2)
    cmax = min(shape[1], ymax + box_size[1]/2)
    rmin = max(0, xmin - box_size[0]/2)
    rmax = min(shape[0], xmax + box_size[0]/2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]

    # It seems that I cannot memmap the same file multiple times without errors
    with fits.open(filename, memmap=False) as a:
        if NAXIS ==2:
            data = a[0].section[rmin:rmax,cmin:cmax]
        elif NAXIS == 3:
            data = a[0].section[0,rmin:rmax,cmin:cmax]
        elif NAXIS ==4:
            data = a[0].section[0,0,rmin:rmax,cmin:cmax]
        else:
            logging.error("Too many NAXIS for me {0}".format(NAXIS))
            logging.error("fix your file to be more sane")
            sys.exit(1)

    # x/y min/max should refer to indices into data
    # this is the region over which we want to operate
    ymin -= cmin
    ymax -= cmin
    xmin -= rmin
    xmax -= rmin

    #start a new RunningPercentile class
    rp = RP()
    def locations(step_size,xmin,xmax,ymin,ymax):
        """
        Generator function to iterate over a grid of x,y coords
        operates only within the given bounds
        Returns:
        x,y,previous_x,previous_y
        """

        xvals = range(xmin,xmax,step_size[0])
        if xvals[-1]!=xmax:
            xvals.append(xmax)
        yvals = range(ymin,ymax,step_size[1])
        if yvals[-1]!=ymax:
            yvals.append(ymax)
        #initial data
        px,py=xvals[0],yvals[0]
        i=1
        for y in yvals:
            for x in xvals[::i]:
                yield x,y,px,py
                px,py=x,y
            i*=-1 #change x direction

    def box(x,y):
        """
        calculate the boundaries of the box centered at x,y
        with size = box_size
        """
        x_min = max(0,x-box_size[0]/2)
        x_max = min(data.shape[0]-1,x+box_size[0]/2)
        y_min = max(0,y-box_size[1]/2)
        y_max = min(data.shape[1]-1,y+box_size[1]/2)
        return x_min,x_max,y_min,y_max

    bkg_points = []
    rms_points = []
    bkg_values = []
    rms_values = []

    # intialise the rp with our first box worth of data
    x_min,x_max,y_min,y_max = box(xmin,ymin)
    new = data[x_min:x_max,y_min:y_max].ravel()
    rp.add(new)

    for x,y,px,py in locations(step_size,xmin,xmax,ymin,ymax):
        x_min,x_max,y_min,y_max = box(x,y)
        px_min,px_max,py_min,py_max = box(px,py)
        old=[]
        new=[]
        #we only move in one direction at a time, but don't know which
        if (x_min>px_min) or (x_max>px_max):
            #down
            if x_min != px_min:
                old = data[min(px_min,x_min):max(px_min,x_min),y_min:y_max].ravel()
            if x_max != px_max:
                new = data[min(px_max,x_max):max(px_max,x_max),y_min:y_max].ravel()
        elif (x_min<px_min) or (x_max<px_max):
            #up
            if x_min != px_min:
                new = data[min(px_min,x_min):max(px_min,x_min),y_min:y_max].ravel()
            if x_max != px_max:
                old = data[min(px_max,x_max):max(px_max,x_max),y_min:y_max].ravel()
        else: # x's have not changed
            #we are moving right
            if y_min != py_min:
                old = data[x_min:x_max,min(py_min,y_min):max(py_min,y_min)].ravel()
            if y_max != py_max:
                new = data[x_min:x_max,min(py_max,y_max):max(py_max,y_max)].ravel()
        rp.add(new)
        rp.sub(old)
        p0,p25,p50,p75,p100 = rp.score()

        if p50 is not None:
            bkg_points.append((x+rmin,y+cmin)) #the coords need to be indices into the larger array
            bkg_values.append(p50)
        if (p75 is not None) and (p25 is not None):
            rms_points.append((x+rmin,y+cmin))
            rms_values.append((p75-p25)/1.34896)

    ymin,ymax,xmin,xmax = region
    # check if we have been passed some shared memory references
    # and do the interpolation if we have
    # otherwise pass back our coords and lists so that interpolation can be done elsewhere
    if ibkg is not None and irms is not None:
        gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
        ifunc = LinearNDInterpolator(rms_points ,rms_values)
        interpolated_rms = ifunc((gx,gy))
        with irms.get_lock():
            for i,row in enumerate(interpolated_rms):
                start_idx = np.ravel_multi_index((xmin + i,ymin), shape)
                end_idx = start_idx + len(row)
                irms[start_idx:end_idx] = row

        ifunc = LinearNDInterpolator(bkg_points ,bkg_values)
        interpolated_bkg = ifunc((gx,gy))
        with ibkg.get_lock():
            for i,row in enumerate(interpolated_bkg):
                start_idx = np.ravel_multi_index((xmin + i,ymin), shape)
                end_idx = start_idx + len(row)
                ibkg[start_idx:end_idx] = row
        logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        return
    else:
        logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        return xmin, xmax, ymin, ymax, bkg_points, bkg_values, rms_points, rms_values


def dummy_filter(filename, region, step_size, box_size, shape, ibkg=None, irms=None):
    """
    Perform a running filter over a region within a file.
    The region can be a sub set of the data within the file - only the useful data will be loaded.

    :param filename: File from which to extract data
    :param region: [xmin,xmax,ymin,ymax] indices over which we are to operate
    :param step_size: amount to move filtering box each iteration
    :param box_size: Size of filtering box
    :return: xmin, xmax, ymin, ymax, bkg_points, bkg_values, rms_points, rms_values
    """

    # Caveat emptor: The code that follows is very difficult to read.
    # xmax is not x_max, and x,y actually should be y,x
    # TODO: fix the code below so that the above comment can be removed

    ymin,ymax,xmin,xmax = region

    logging.debug('{0}x{1},{2}x{3} starting at {4} - dummy'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    cmin = max(0, ymin - box_size[1]/2)
    cmax = min(shape[1], ymax + box_size[1]/2)
    rmin = max(0, xmin - box_size[0]/2)
    rmax = min(shape[0], xmax + box_size[0]/2)

    # x/y min/max should refer to indices into data
    # this is the region over which we want to operate
    ymin -= cmin
    ymax -= cmin
    xmin -= rmin
    xmax -= rmin

    #start a new RunningPercentile class
    def locations(step_size,xmin,xmax,ymin,ymax):
        """
        Generator function to iterate over a grid of x,y coords
        operates only within the given bounds
        Returns:
        x,y,previous_x,previous_y
        """

        xvals = range(xmin,xmax,step_size[0])
        if xvals[-1]!=xmax:
            xvals.append(xmax)
        yvals = range(ymin,ymax,step_size[1])
        if yvals[-1]!=ymax:
            yvals.append(ymax)
        #initial data
        px,py=xvals[0],yvals[0]
        i=1
        for y in yvals:
            for x in xvals[::i]:
                yield x,y,px,py
                px,py=x,y
            i*=-1 #change x direction

    def box(x,y):
        """
        calculate the boundaries of the box centered at x,y
        with size = box_size
        """
        x_min = max(0,x-box_size[0]/2)
        x_max = min(data.shape[0]-1,x+box_size[0]/2)
        y_min = max(0,y-box_size[1]/2)
        y_max = min(data.shape[1]-1,y+box_size[1]/2)
        return x_min,x_max,y_min,y_max

    bkg_points = []
    rms_points = []
    bkg_values = []
    rms_values = []

    for x,y,px,py in locations(step_size,xmin,xmax,ymin,ymax):
        bkg_points.append((x+rmin,y+cmin)) #the coords need to be indices into the larger array
        bkg_values.append(region[0])

        rms_points.append((x+rmin,y+cmin))
        rms_values.append(region[2])

    #return our lists, the interpolation will be done on the master node
    #also tell the master node where the data came from - using the original coords
    ymin,ymax,xmin,xmax = region

    # check if we have been passed some shared memory references
    if ibkg is not None and irms is not None:
        gx,gy = np.mgrid[xmin:xmax,ymin:ymax]
        ifunc = LinearNDInterpolator(rms_points ,rms_values)
        interpolated_rms = ifunc((gx,gy))
        with irms.get_lock():
            for i,row in enumerate(interpolated_rms):
                start_idx = np.ravel_multi_index((xmin + i,ymin), shape)
                end_idx = start_idx + len(row)
                #print len(row), len(irms[start_idx:end_idx])
                irms[start_idx:end_idx] = row

        ifunc = LinearNDInterpolator(bkg_points ,bkg_values)
        interpolated_bkg = ifunc((gx,gy))
        with ibkg.get_lock():
            for i,row in enumerate(interpolated_bkg):
                start_idx = np.ravel_multi_index((xmin + i,ymin), shape)
                end_idx = start_idx + len(row)
                #print len(row), len(irms[start_idx:end_idx])
                ibkg[start_idx:end_idx] = row
        logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        return
    else:
        logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        return xmin, xmax, ymin, ymax, bkg_points, bkg_values, rms_points, rms_values


def gen_factors(m, permute=True):
    """
    Generate a list of integer factors for m
    :param m: A positive integer
    :param permute: returns permutations instead of combinations
    :return:
    """
    # convert to int if people have been naughty
    n=int(abs(m))
    # brute force the factors, one of which is always less than sqrt(n)
    for i in xrange(1, int(n**0.5+1)):
        if n % i == 0:
            yield i, n/i
            # yield the reverse pair if it is unique
            if i != n/i and permute:
                yield n/i, i


def optimum_sections(cores, data_shape):
    """
    Choose the best sectioning scheme based on the number of cores available and the shape of the data
    :param cores: Number of available cores
    :param data_shape: Shape of the data as [x,y]
    :return: (nx,ny) the number of divisions in each direction
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
    Masking is done via np.nan values (or any not finite values).
    :param data: A 2d array of data
    :param mask_data: An image of at least 2d, some of which may be nan/blank
    :return: None, data is modified to be np.nan in the places where mask_data is not finite
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


def filter_mc_sharemem(filename, step_size, box_size, cores, shape, fn=sigma_filter):
    """
    Perform a running filter over multiple cores

    :param filename: data file name
    :param step_size: mesh/grid increment in pixels
    :param box_size: size of box over which the filtering is done
    :param cores: number of cores to use
    :param shape: shape of the data array in the file 'filename'
    :param fn: the function which performs the filtering
    :return:
    """

    if cores is None:
        cores = multiprocessing.cpu_count()
    if cores > 1:
        try:
            queue = pprocess.Queue(limit=cores, reuse=0)
            parfilt = queue.manage(pprocess.MakeParallel(fn))
        except AttributeError, e:
            if 'poll' in e.message:
                logging.warn("Your O/S doesn't support select.poll(): Reverting to cores=1")
                cores = 1
            else:
                logging.error("Your system can't seem to make a queue, try using --cores=1")
                raise e
    img_y, img_x = shape
    # initialise some shared memory
    alen = shape[0]*shape[1]
    ibkg = multiprocessing.Array('f', alen)
    irms = multiprocessing.Array('f', alen)

    if cores > 1:
        logging.info("using {0} cores".format(cores))
        nx, ny = optimum_sections(cores, shape)

        # box widths should be multiples of the step_size, and not zero
        width_x = max(img_x/nx/step_size[0], 1) * step_size[0]
        width_y = max(img_y/ny/step_size[1], 1) * step_size[1]

        xstart = width_x
        ystart = width_y
        xend = img_x - img_x % width_x  # the end point of the last "full" box
        yend = img_y - img_y % width_y

        # locations of the box edges
        xmins = [0]
        xmins.extend(range(xstart, xend, width_x))

        xmaxs = [xstart]
        xmaxs.extend(range(xstart+width_x, xend+1, width_x))
        xmaxs[-1] = img_x

        ymins = [0]
        ymins.extend(range(ystart, yend, width_y))

        ymaxs = [ystart]
        ymaxs.extend(range(ystart+width_y, yend+1, width_y))
        ymaxs[-1] = img_y

        for xmin, xmax in zip(xmins, xmaxs):
            for ymin, ymax in zip(ymins, ymaxs):
                region = [xmin, xmax, ymin, ymax]
                parfilt(filename, region, step_size, box_size, shape, ibkg, irms)

        # Need to wait for the queue to finish processing before we continue
        # This requires that we have reuse=0 and makeparallel when we start the queue
        for _ in queue:
            pass


    else:
        # single core we do it all at once
        region = [0, img_x, 0, img_y]
        fn(filename, region, step_size, box_size, shape, ibkg, irms)
    # reshape our 1d arrays back into a 2d image
    logging.debug("reshaping bkg")
    interpolated_bkg = np.reshape(ibkg, shape)
    logging.debug("reshaping rms")
    interpolated_rms = np.reshape(irms, shape)

    if cores > 1:
        del queue, parfilt
    return interpolated_bkg, interpolated_rms


def filter_image(im_name, out_base, step_size=None, box_size=None, twopass=False, cores=None, mask=True, compressed=False, running=False):
    """

    :param im_name:
    :param out_base:
    :param step_size:
    :param box_size:
    :param twopass:
    :param cores:
    :param mask:
    :param compressed:
    :param running: use the running percentiles filter.
    :return:
    """

    if running:
        func = running_filter
    else:
        func = sigma_filter

    header = fits.getheader(im_name)
    shape = (header['NAXIS2'],header['NAXIS1'])

    if step_size is None:
        if 'BMAJ' in header and 'BMIN' in header:
            beam_size = np.sqrt(abs(header['BMAJ']*header['BMIN']))
            if 'CDELT1' in header:
                pix_scale = np.sqrt(abs(header['CDELT1']*header['CDELT2']))
            elif 'CD1_1' in header:
                pix_scale = np.sqrt(abs(header['CD1_1']*header['CD2_2']))
                if header['CD1_2'] != 0 or header['CD2_1']!=0:
                    logging.warn("CD1_2 and/or CD2_1 are non-zero and I don't know what to do with them")
                    logging.warn("Ingoring them")
            else:
                logging.warn("Cannot determine pixel scale, assuming 4 pixels per beam")
                pix_scale = beam_size/4.
            # default to 4x the synthesized beam width
            step_size = int(np.ceil(4*beam_size/pix_scale))
        else:
            logging.info("BMAJ and/or BMIN not in fits header.")
            logging.info("Assuming 4 pix/beam, so we have step_size = 16 pixels")
            step_size = 16
        step_size = (step_size,step_size)

    if box_size is None:
        # default to 6x the step size so we have ~ 30beams
        box_size = (step_size[0]*6,step_size[1]*6)

    if compressed:
        if not step_size[0] == step_size[1]:
            step_size = (min(step_size),min(step_size))
            logging.info("Changing grid to be {0} so we can compress the output".format(step_size))

    logging.info("using grid_size {0}, box_size {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(shape))
    bkg, rms = filter_mc_sharemem(im_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, fn=func)
    logging.info("done")

    if twopass:
        # TODO: check what this does for our memory usage
        tempfile = NamedTemporaryFile(delete=False)
        data = fits.getdata(im_name) - bkg
        header = fits.getheader(im_name)
        write_fits(data, header, tempfile)
        tempfile.close()
        temp_name = tempfile.name
        del data, header, tempfile
        logging.info("running second pass to get a better rms")
        _, rms = filter_mc_sharemem(temp_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, fn=func)
        os.remove(temp_name)

    bkg_out = '_'.join([os.path.expanduser(out_base), 'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base), 'rms.fits'])

    # force float 32s to avoid bloated files
    bkg = np.array(bkg, dtype=np.float32)
    rms = np.array(rms, dtype=np.float32)

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
# Alternate Filters
# Used only for testing algorithm speeds, not really useful
###
def scipy_filter(im_name, out_base, step_size, box_size, cores=None):
    from scipy.ndimage.filters import generic_filter
    from scipy.stats import nanmedian,nanstd,scoreatpercentile

    fits,data = load_image(im_name)

    if step_size is None:
        pix_scale = np.sqrt(abs(fits[0].header['CDELT1']*fits[0].header['CDELT2']))
        beam_size = np.sqrt(abs(fits[0].header['BMAJ']*fits[0].header['BMIN']))
        #default to 4x the synthesized beam width               
        step_size = int(np.ceil(4*beam_size/pix_scale))
        step_size = (step_size,step_size)

    if box_size is None:
        #default to 5x the step size
        box_size = (step_size[0]*5,step_size[1]*5)

    logging.info("using grid {0}, box {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(data.shape))
    logging.info("with scipy generic filter median/std")
    #scipy can't handle nan values when using score at percentile
    def iqrms(x):
        d=x[np.isfinite(x)]
        if len(d)<2:
            return np.nan
        a=scoreatpercentile(d,[75,25])
        return  (a[0]-a[1])/1.34896
    def median(x):
        d=x[np.isfinite(x)]
        if len(d)<2:
            return np.nan
        a=scoreatpercentile(d,50)
        return a

    bkg = generic_filter(data,median,size=box_size)
    rms = generic_filter(data-bkg,iqrms,size=box_size)

    bkg_out = '_'.join([os.path.expanduser(out_base),'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base),'rms.fits'])
    #masking
    mask = np.isnan(data)
    bkg[mask]=np.NaN
    rms[mask]=np.NaN

    save_image(fits,bkg,bkg_out)
    save_image(fits,rms,rms_out)

###
# Helper functions
###
def load_image(im_name):
    """
    Generic helper function to load a fits file
    """
    try:
        fitsfile = fits.open(im_name)
    except IOError, e:
        if "END" in e.message:
            logging.warn(e.message)
            logging.warn("trying to ignore this, but you should really fix it")
            fitsfile = fits.open(im_name, ignore_missing_end=True)

    data = fitsfile[0].data
    if fitsfile[0].header['NAXIS']>2:
        data = data.squeeze()  # remove axes with length 1
    logging.info("loaded {0}".format(im_name))
    return fitsfile, data


def write_fits(data, header, file_name):
    """

    :param data:
    :param file_name:
    :return:
    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_name, clobber=True)
    logging.info("Wrote {0}".format(file_name))


def save_image(hdu, data, im_name):
    """
    Generic helper function to save a fits file with a given name/header
    This function modifies the fits object!
    """
    hdu[0].data = data
    hdu[0].header['HISTORY']='BANE {0}-({1})'.format(__version__,__date__)
    try:
        hdu.writeto(im_name,clobber=True)
    except hdu.verify.VerifyError,e:
        if "DATAMAX" in e.message or "DATAMIN" in e.message:
            logging.warn(e.message)
            logging.warn("I will fix this but it will cause some programs to break")
            hdu.writeto(im_name,clobber=True,output_verify="silentfix")
    logging.info("wrote {0}".format(im_name))
    return


# command line version of this program runs from here.
if __name__=="__main__":
    usage = "usage: %prog [options] FileName.fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--out",dest='out_base',
                      help="Basename for output images default: FileName_{bkg,rms}.fits")
    parser.add_option('--grid',dest='step_size',type='int',nargs=2,
                      help='The [x,y] size of the grid to use. Default = ~4* beam size square.')
    parser.add_option('--box',dest='box_size',type='int',nargs=2,
                      help='The [x,y] size of the box over which the rms/bkg is calculated. Default = 5*grid.')
    parser.add_option('--cores',dest='cores',type='int',
                      help='Number of cores to use. Default = all available.')
    parser.add_option('--onepass',dest='twopass',action='store_false', help='the opposite of twopass. default=False')
    parser.add_option('--twopass',dest='twopass',action='store_true',
                      help='Calculate the bkg and rms in a two passes instead of one. (when the bkg changes rapidly)')
    parser.add_option('--nomask', dest='mask', action='store_false', default=True,
                      help="Don't mask the output array [default = mask]")
    parser.add_option('--noclobber', dest='clobber',action='store_false', default=True,
                      help="Don't run if output files already exist. Default is to run+overwrite.")
    parser.add_option('--scipy',dest='usescipy',action='store_true',
                      help='Use scipy generic filter instead of the running percentile filter. (for testing/timing)')
    parser.add_option('--RP', dest='running', action='store_true', default=False,
                      help='Use the old/slower running percentiles filter')
    parser.add_option('--debug',dest='debug',action='store_true',help='debug mode, default=False')
    parser.add_option('--compress', dest='compress', action='store_true',default=False,
                      help='Produce a compressed output file.')
    parser.set_defaults(out_base=None,step_size=None,box_size=None,twopass=True,cores=None,usescipy=False,debug=False)
    (options, args) = parser.parse_args()

    logging_level = logging.DEBUG if options.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is BANE {0}-({1})".format(__version__,__date__))
    if len(args)<1:
        parser.print_help()
        sys.exit()
    else:
        filename = args[0]
    if not os.path.exists(filename):
        logging.error("File not found: {0} ".format(filename))
        sys.exit(1)

    if options.out_base is None:
        options.out_base = os.path.splitext(filename)[0]

    if not options.clobber:
        bkgout, rmsout = options.out_base+'_bkg.fits', options.out_base+'_rms.fits'
        if os.path.exists(bkgout) and os.path.exists(rmsout):
            logging.error("{0} and {1} exist and you said noclobber".format(bkgout,rmsout))
            logging.error("Not running")
            sys.exit(1)


    if options.usescipy:
        scipy_filter(im_name=filename,out_base=options.out_base,step_size=options.step_size,box_size=options.box_size,cores=options.cores)
    else:
        filter_image(im_name=filename, out_base=options.out_base, step_size=options.step_size,
                     box_size=options.box_size, twopass=options.twopass, cores=options.cores,
                     mask=options.mask, compressed=options.compress, running=options.running)

