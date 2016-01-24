#! /usr/bin/env python
"""
 Tool for making residual images with Aegean tables as input
"""
__author__ = 'paulhancock'

import sys
import numpy as np
import math

from astropy.io import fits
from AegeanTools import catalogs, wcs_helpers, fitting

# global constants
FWHM2CC = 1 / (2 * math.sqrt(2 * math.log(2)))

def load_sources(filename):
    """
    Open a file, read contents, return a list of all the sources in that file.
    @param filename:
    @return: list of OutputSource objects
    """
    return catalogs.table_to_source_list(catalogs.load_table(filename))


def make_model(sources, shape, wcshelper):
    """

    @param source:
    @param data:
    @param wcs:
    @return:
    """

    # Model array
    m = np.zeros(shape,dtype=np.float32)
    factor = 5

    for src in sources:
        xo, yo, sx, sy, theta = wcshelper.sky2pix_ellipse([src.ra, src.dec], src.a/3600, src.b/3600, src.pa)
        phi = np.radians(theta)

        # pixels over which this model is calculated
        xoff = abs(factor*(sx*np.cos(phi)+ sy*np.sin(phi)))
        xmin = xo - xoff
        xmax = xo + xoff
        yoff = abs(factor*(sx*np.sin(phi)+ sy*np.cos(phi)))
        ymin = yo - yoff
        ymax = yo + yoff

        #clip to the image size
        ymin = max(np.floor(ymin),0)
        ymax = min(np.ceil(ymax), shape[1])

        xmin = max(np.floor(xmin),0)
        xmax = min(np.ceil(xmax), shape[0])

        if False:
            print "Source ({0},{1})".format(src.island, src.source)
            print " xo,yo, sx, sy, theta, phi", xo,yo, sx, sy, theta, phi
            print " xoff, yoff", xoff, yoff
            print " xmin, xmax, ymin, ymax", xmin, xmax, ymin, ymax

        #positions for which we want to make the model
        x, y = np.mgrid[xmin:xmax,ymin:ymax]
        x = map(int,x.ravel())
        y = map(int,y.ravel())

        # TODO: understand why xo/yo -1 is needed
        model = fitting.elliptical_gaussian(x, y, src.peak_flux, xo-1, yo-1, sx*FWHM2CC, sy*FWHM2CC, theta)
        m[x, y] += model
    return m

def make_residual(fitsfile, catalog, outfile):
    """

    @param fitsfile:
    @param catalog:
    @param outfile:
    @return:
    """
    source_list = load_sources(catalog)
    hdulist = fits.open(fitsfile)
    data =hdulist[0].data
    header = hdulist[0].header

    wcshelper = wcs_helpers.WCSHelper.from_header(header)

    model = make_model(source_list, data.shape, wcshelper)

    residual = data - model

    hdulist[0].data = residual
    hdulist.writeto(outfile, clobber=True)
    print "wrote {0}".format(outfile)
    hdulist[0].data = model
    hdulist.writeto('model.fits', clobber=True)
    print "wrote model.fits"
    return






if __name__ == "__main__":
    if len(sys.argv)>3:
        fitsfile, catalog, outfile = sys.argv[-3:]
    else:
        print "usage: python residual.py <fitsfile> <catalog> <outfits>"
        sys.exit(1)

    print "Using {0} and {1} to make {2}".format(fitsfile, catalog, outfile)

    make_residual(fitsfile, catalog, outfile)

