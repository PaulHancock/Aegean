#! /usr/bin/env python

"""
A module to allow fits files to be shrunk in size using decimation, and to be
grown in size using interpolation.

@author: Paul Hancock

Created:
3rd Feb 2015
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
import logging


def load_file_or_hdu(filename):
    """
    Load a file from disk and return an HDUList
    If filename is already an HDUList return that instead
    :param filename: Filename or HDUList
    :return: HDUList
    """
    if isinstance(filename, fits.HDUList):
        hdulist = filename
    else:
        try:
            hdulist = fits.open(filename)
        except IOError, e:
            if "END" in e.message:
                logging.warn(e.message)
                logging.warn("trying to ignore this, but you should really fix it")
                hdulist = fits.open(filename, ignore_missing_end=True)
            else:
                raise e
    return hdulist


def compress(datafile, factor, outfile=None):
    """

    :param datafile: Filename or HDUList (hdulist will be modified)
    :param factor: factor to be reduced
    :param outfile: filename to write (default=None, don't write to file)
    :return: An HDUlist of the reduced data
    """
    if not (factor > 0 and isinstance(factor, int)):
        logging.error("factor must be a positive integer")
        return None

    hdulist = load_file_or_hdu(datafile)

    header = hdulist[0].header
    data = np.squeeze(hdulist[0].data)
    cx, cy = data.shape[0], data.shape[1]

    nx = cx / factor
    ny = cy / factor
    # check to see if we will have some residual data points
    lcx = cx % factor
    lcy = cy % factor
    if lcx > 0:
        nx += 1
    if lcy > 0:
        ny += 1
    # decimate the data
    new_data = np.empty((nx + 1, ny + 1))
    new_data[:nx, :ny] = data[::factor, ::factor]
    # copy the last row/col across
    new_data[-1, :ny] = data[-1, ::factor]
    new_data[:nx, -1] = data[::factor, -1]
    new_data[-1, -1] = data[-1, -1]

    # TODO: Figure out what to do when CD2_1 and CD1_2 are non-zero
    if 'CDELT1' in header:
        header['CDELT1'] *= factor
    elif 'CD1_1' in header:
        header['CD1_1'] *= factor
    else:
        logging.error("Error: Can't find CDELT1 or CD1_1")
        return None
    if 'CDELT2' in header:
        header['CDELT2'] *= factor
    elif "CD2_2" in header:
        header['CD2_2'] *= factor
    else:
        logging.error("Error: Can't find CDELT2 or CD2_2")
        return None
    # Move the reference pixel so that the WCS is correct
    header['CRPIX1'] = (header['CRPIX1'] + factor - 1) / factor
    header['CRPIX2'] = (header['CRPIX2'] + factor - 1) / factor

    # Update the header so that we can do the correct interpolation later on
    header['BN_CFAC'] = (factor, "Compression factor (grid size) used by BANE")
    header['BN_NPX1'] = (header['NAXIS1'], 'original NAXIS1 value')
    header['BN_NPX2'] = (header['NAXIS2'], 'original NAXIS2 value')
    header['BN_RPX1'] = (lcx, 'Residual on axis 1')
    header['BN_RPX2'] = (lcy, 'Residual on axis 2')
    header['HISTORY'] = "Compressed by a factor of {0}".format(factor)

    # save the changes
    hdulist[0].data = np.array(new_data, dtype=np.float32)
    hdulist[0].header = header
    if outfile is not None:
        hdulist.writeto(outfile, clobber=True)
        logging.info("Wrote: {0}".format(outfile))
    return hdulist


def expand(datafile, outfile=None, method='linear'):
    """
    Expand and interpolate the given data file using the given method.
    Datafile can be a filename or an HDUList
    interpolation is carried out by scipy.interpolate.griddata so method can be any valid method
    accepted by that function.

    :param datafile: filename or HDUList of file to work on
    :param outfile: filename to write to (default = None)
    :param method: interpolation method (default='linear')
    :return: HDUList of the expanded data
    """
    hdulist = load_file_or_hdu(datafile)

    header = hdulist[0].header
    data = hdulist[0].data
    # Check for the required key words, only expand if they exist
    if not all(a in header for a in ['BN_CFAC', 'BN_NPX1', 'BN_NPX2', 'BN_RPX1', 'BN_RPX2']):
        return hdulist

    factor = header['BN_CFAC']
    (gx, gy) = np.mgrid[0:header['BN_NPX2'], 0:header['BN_NPX1']]
    # Extract the data and create the array of indices
    values = np.ravel(data)
    grid = np.indices(data.shape)
    # fix the last column of the grid to account for residuals
    lcx = header['BN_RPX2']
    lcy = header['BN_RPX1']

    grid[0, :] += lcx/factor
    grid[1, :] += lcy/factor
    grid  *= factor
    points = zip(np.ravel(grid[0]), np.ravel(grid[1]))

    # Do the interpolation
    hdulist[0].data = np.array(griddata(points, values, (gx, gy), method=method), dtype=np.float32)

    # update the fits keywords so that the WCS is correct
    header['CRPIX1'] = (header['CRPIX1'] - 1) * factor + 1
    header['CRPIX2'] = (header['CRPIX2'] - 1) * factor + 1

    if 'CDELT1' in header:
        header['CDELT1'] /= factor
    elif 'CD1_1' in header:
        header['CD1_1'] /= factor
    else:
        logging.error("Error: Can't find CD1_1 or CDELT1")
        return None

    if 'CDELT2' in header:
        header['CDELT2'] /= factor
    elif "CD2_2" in header:
        header['CD2_2'] /= factor
    else:
        logging.error("Error: Can't find CDELT2 or CD2_2")
        return None

    header['BN_INTP'] = (method, 'BANE interpolation method')
    header['HISTORY'] = 'Expanded by factor {0}'.format(factor)

    # don't need these any more so delete them.
    del header['BN_CFAC'], header['BN_NPX1'], header['BN_NPX2'], header['BN_RPX1'], header['BN_RPX2']
    hdulist[0].header = header
    if outfile is not None:
        hdulist.writeto(outfile, clobber=True)
        logging.info("Wrote: {0}".format(outfile))
    return hdulist

if __name__ == "__main__":
    compress("Test/Images/1904-66_AIT.fits", factor=7, outfile='test.fits')
    expand('test.fits', outfile='test2.fits', method='linear')
