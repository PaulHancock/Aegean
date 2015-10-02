#! /usr/bin/env python

__author__ = 'Paul Hancock'
__version__ = 'v1.1'
__date__ = '2015-05-19'

import os
import sys
import copy
import logging
import argparse
import numpy as np
from math import floor

import astropy
import astropy.wcs
from astropy.io import fits

def floor2(a):
    return map(lambda x: int(floor(x)),a)


def section(filename, factor=(2,2),outdir=''):
    """

    :param filename:
    :param factor:
    :return:
    """
    init_header = fits.getheader(filename)
    shape = init_header["NAXIS2"],init_header["NAXIS1"]
    boundaries = []
    buffer = 0.1*shape[0]/factor[0], 0.1*shape[1]/factor[1]
    ysize = shape[0]/factor[0]
    xsize = shape[1]/factor[1]
    yedges = range(0,shape[0]+shape[0]%ysize,ysize)
    xedges = range(0,shape[1]+shape[1]%xsize,xsize)

    for i in zip(yedges[:-1],yedges[1:]):
        ylims = floor2((max(i[0]-buffer[0],0), min(i[1]+buffer[0],shape[0])))
        for j in zip(xedges[:-1],xedges[1:]):
            xlims = floor2((max(j[0]-buffer[1],0), min(j[1]+buffer[1],shape[1])))
            boundaries.append((ylims,xlims))

    for i,((ymin,ymax),(xmin,xmax)) in enumerate(boundaries):
        with fits.open(filename, memmap=False) as hdu:
            naxis = hdu[0].header["NAXIS"]
            if naxis==2:
                new_data = hdu[0].section[ymin:ymax,xmin:xmax]
            elif naxis==3:
                new_data = hdu[0].section[0,ymin:ymax,xmin:xmax]
            elif naxis==4:
                new_data = hdu[0].section[0,0,ymin:ymax,xmin:xmax]
            else:
                logging.critical("Your image has too many axes {0}".format(naxis))
                sys.exit()
        new_filename = filename.replace('.fits','_sec{0:02d}.fits'.format(i))
        new_filename = os.path.join(outdir,new_filename)
        logging.info("{0} -> {1}".format(((ymin,ymax),(xmin,xmax)),new_filename))

        # fix the header
        new_header = copy.deepcopy(init_header)
        new_header['CRPIX1'] -= xmin
        new_header['CRPIX2'] -= ymin

        # return the data to the original number of dimensions
        new_shape = [1 for i in range(naxis-2)]+list(new_data.shape)
        new_data = np.reshape(new_data,new_shape)
        
        # save the data
        hdulist = fits.HDUList([fits.PrimaryHDU(data=new_data, header=new_header)])
        hdulist.writeto(new_filename,clobber=True)
        logging.info("Wrote {0}".format(new_filename))


def sub_image(filename, box, outfile=None):
    """

    :param filename:
    :param box:
    :return:
    """
    hdulist = fits.open(filename,memmap=False)
    header = hdulist[0].header
    shape = header["NAXIS2"],header["NAXIS1"]
    wcs = astropy.wcs.WCS(header)

    # figure out the corners of the image in pixel coords
    ramin,ramax,decmin,decmax = box
    tl = (ramax, decmax)
    tr = (ramin, decmax)
    bl = (ramax, decmin)
    br = (ramin, decmin)
    corners = [tl,tr,bl,br]
    corners_pix = wcs.wcs_world2pix(corners,1)
    xmax,ymax = map(lambda x: int(floor(x)), np.max(corners_pix, axis=0))
    xmin,ymin = map(lambda x: int(floor(x)), np.min(corners_pix, axis=0))

    ymax = np.clip(ymax,0,shape[0])
    ymin = np.clip(ymin,0,shape[0])
    xmax = np.clip(xmax,0,shape[1])
    xmin = np.clip(xmin,0,shape[1])
    if ymax<=ymin or xmax<=xmin:
        print "Requested region not within file"
        return
    # get the required region
    data = hdulist[0].section[ymin:ymax,xmin:xmax]
    if outfile is None:
        outfile = filename.lower().replace('.fits','_sub.fits')
    # redo the header so that wcs is correct
    header['CRPIX1'] -= xmin
    header['CRPIX2'] -= ymin
    hdulist[0].data = data
    hdulist.writeto(outfile,clobber=True)
    logging.info("Write {0}".format(outfile))



def rejoin(filelist, outfile):
    """
    :param filelist:
    :param outfile:
    :return:
    """
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fitsfile', type=str, help='File for splitting. default: none')
    group1 = parser.add_argument_group("options for extracting sub regions")
    group1.add_argument("--outdir",dest='out_dir', type=str, default='./',metavar="DIR",
                        help="Directory for output images default: ./")
    group1.add_argument('--factor',dest='factor',type=int,nargs=2, default=None, metavar=("N","M"),
                         help='Cut the image into a set of NxM sub images')
    group1.add_argument('--cutout',dest='cutout',type=float, nargs=4, default=None, metavar=("RAmin", "RAmax", "DECmin", "DECmax"),
                        help="The boundaries of the region to cut out")
    group1.add_argument('--out', dest='outfile', type=str,default=None,metavar="Filename.fits",
                        help="A model for the output filename. If more than one file is output then all files will have a number appended to them.")
    results = parser.parse_args()

    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    #logging.info("This is BANE {0}-({1})".format(__version__,__date__))

    if not os.path.exists(results.fitsfile):
        logging.error("File not found: {0} ".format(results.fitsfile))
        sys.exit(1)

    if results.cutout is not None:
        sub_image(results.fitsfile,results.cutout,results.outfile)

    if results.factor is not None:
        section(results.fitsfile,results.factor,results.out_dir)