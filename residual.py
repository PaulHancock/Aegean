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

#global constants
fwhm2cc = 1 / (2 * math.sqrt(2 * math.log(2)))
cc2fwhm = (2 * math.sqrt(2 * math.log(2)))

if __name__ == "__main__":
    if len(sys.argv)>3:
        fitsfile, catalog, outfile = sys.argv[-3:]
    else:
        print "usage: python residual.py <fitsfile> <catalog> <outfits>"
        sys.exit(1)

    print "Using {0} and {1} to make {2}".format(fitsfile, catalog, outfile)

    srclist = catalogs.table_to_source_list(catalogs.load_table(catalog))

    hdulist = fits.open(fitsfile)
    data =hdulist[0].data
    header = hdulist[0].header

    wcshelper = wcs_helpers.WCSHelper.from_header(header)

    x, y = np.where(np.isfinite(data))

    m = data*0

    for src in srclist:
        amp = src.peak_flux
        xo,yo = wcshelper.sky2pix([src.ra,src.dec])
        _,_,sx,theta = wcshelper.sky2pix_vec([src.ra,src.dec],src.a/3600,src.pa)
        _,_,sy,_ = wcshelper.sky2pix_vec([src.ra,src.dec],src.b/3600,src.pa+90)
        model = fitting.elliptical_gaussian(x,y,amp,xo-1,yo-1,sx*fwhm2cc,sy*fwhm2cc,theta)
        m[x,y] += model

    hdulist[0].data = data - m
    hdulist.writeto(outfile,clobber=True)
