#! /usr/bin/env python

# standard imports
import numpy as np
import sys
import os
from optparse import OptionParser
from time import gmtime, strftime
import logging
import copy
from tempfile import NamedTemporaryFile

from scipy.interpolate import LinearNDInterpolator
from astropy.io import fits


# Aegean tools
from AegeanTools.fits_interp import compress

import multiprocessing

__author__ = 'Paul Hancock'
__version__ = 'v1.4.1'
__date__ = '2016-02-15'


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


def sf2(args):
    """
    Wrapper for sigma_filter
    """
    return sigma_filter(*args)


def sigma_filter(filename, region, step_size, box_size, shape, dobkg=True):
    """
    Calculated the rms [and background] for a sub region of an image. Save the resulting calculations
    into shared memory - irms [and ibkg].
    :param filename: Fits file to open
    :param region: Region within fits file that is to be processed
    :param step_size: The filtering step size
    :param box_size: The size of the box over which the filter is applied (each step)
    :param shape: The shape of the fits image
    :param dobkg: True = do background calculation.
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
    bkg_values = []
    rms_points = []
    rms_values = []

    for x, y in locations(step_size, xmin, xmax, ymin, ymax):
        x_min, x_max, y_min, y_max = box(x, y)
        new = data[x_min:x_max, y_min:y_max]
        new = np.ravel(new)
        new = sigmaclip(new, 3, 3)
        # If we are left with (or started with) no data, then just move on
        if len(new)<1:
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
        gx, gy = np.mgrid[xmin:xmax, ymin:ymax]
        if len(bkg_points)>1:
            logging.debug("Interpolating bkg")
            ifunc = LinearNDInterpolator(bkg_points, bkg_values)
            interpolated_bkg = np.array(ifunc((gx, gy)), dtype=np.float32)
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
    del ifunc
    return


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


def filter_mc_sharemem(filename, step_size, box_size, cores, shape, dobkg=True):
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

    args = []
    for xmin, xmax in zip(xmins, xmaxs):
        for ymin, ymax in zip(ymins, ymaxs):
            region = [xmin, xmax, ymin, ymax]
            args.append((filename, region, step_size, box_size, shape, dobkg))

    pool = multiprocessing.Pool(processes=cores)
    pool.map(sf2, args)
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
    bkg, rms = filter_mc_sharemem(im_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape)
    logging.info("done")

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
        _, rms = filter_mc_sharemem(temp_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, dobkg=False)
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
    hdu[0].header['HISTORY']='BANE {0}-({1})'.format(__version__, __date__)
    try:
        hdu.writeto(im_name, clobber=True)
    except hdu.verify.VerifyError,e:
        if "DATAMAX" in e.message or "DATAMIN" in e.message:
            logging.warn(e.message)
            logging.warn("I will fix this but it will cause some programs to break")
            hdu.writeto(im_name, clobber=True, output_verify="silentfix")
    logging.info("wrote {0}".format(im_name))
    return


# command line version of this program runs from here.
if __name__=="__main__":
    usage = "usage: %prog [options] FileName.fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--out", dest='out_base',
                      help="Basename for output images default: FileName_{bkg,rms}.fits")
    parser.add_option('--grid', dest='step_size', type='int', nargs=2,
                      help='The [x,y] size of the grid to use. Default = ~4* beam size square.')
    parser.add_option('--box', dest='box_size', type='int', nargs=2,
                      help='The [x,y] size of the box over which the rms/bkg is calculated. Default = 5*grid.')
    parser.add_option('--cores', dest='cores', type='int',
                      help='Number of cores to use. Default = all available.')
    parser.add_option('--onepass', dest='twopass', action='store_false', help='the opposite of twopass. default=False')
    parser.add_option('--twopass', dest='twopass', action='store_true',
                      help='Calculate the bkg and rms in a two passes instead of one. (when the bkg changes rapidly)')
    parser.add_option('--nomask', dest='mask', action='store_false', default=True,
                      help="Don't mask the output array [default = mask]")
    parser.add_option('--noclobber', dest='clobber', action='store_false', default=True,
                      help="Don't run if output files already exist. Default is to run+overwrite.")
    parser.add_option('--debug', dest='debug', action='store_true', help='debug mode, default=False')
    parser.add_option('--compress', dest='compress', action='store_true', default=False,
                      help='Produce a compressed output file.')
    parser.set_defaults(out_base=None, step_size=None, box_size=None, twopass=True, cores=None, usescipy=False, debug=False)
    (options, args) = parser.parse_args()

    logging_level = logging.DEBUG if options.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is BANE {0}-({1})".format(__version__, __date__))
    if len(args) < 1:
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
            logging.error("{0} and {1} exist and you said noclobber".format(bkgout, rmsout))
            logging.error("Not running")
            sys.exit(1)

    filter_image(im_name=filename, out_base=options.out_base, step_size=options.step_size,
                 box_size=options.box_size, twopass=options.twopass, cores=options.cores,
                 mask=options.mask, compressed=options.compress)

