#! /usr/bin/env python

__author__ = 'Paul Hancock'
__version__ = 'v1.1'
__date__ = '2015-05-19'

import os
import sys
import copy
import logging
from optparse import OptionParser
from math import floor

import astropy
from astropy.io import fits

def floor2(a):
    return map(lambda x: int(floor(x)),a)


def section(filename, factor=(2,2)):
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
    yedges = range(0,shape[0],ysize)
    if yedges[-1] != shape[0]:
        yedges.append(shape[0])
    xedges = range(0,shape[1],xsize)
    if xedges[-1] != shape[1]:
        xedges.append(shape[1])

    for i in zip(yedges[:-1],yedges[1:]):
        ylims = floor2((max(i[0]-buffer[0],0), min(i[1]+buffer[0],shape[0])))
        for j in zip(xedges[:-1],xedges[1:]):
            xlims = floor2((max(j[0]-buffer[1],0), min(j[1]+buffer[1],shape[1])))
            boundaries.append((ylims,xlims))

    for i,((ymin,ymax),(xmin,xmax)) in enumerate(boundaries):
        new_data = fits.open(filename)[0].section[ymin:ymax,xmin:xmax]
        new_filename = filename.replace('.fits','_sec{0:02d}.fits'.format(i))
        logging.info("{0} -> {1}".format(((ymin,ymax),(xmin,xmax)),new_filename))

        # fix the header
        new_header = copy.deepcopy(init_header)
        new_header['CRPIX1'] -= xmin
        new_header['CRPIX2'] -= ymin

        hdulist = fits.HDUList([fits.PrimaryHDU(data=new_data, header=new_header)])
        hdulist.writeto(new_filename,clobber=True)
        logging.info("Wrote {0}".format(new_filename))





def rejoin(filelist, outfile):
    """
    :param filelist:
    :param outfile:
    :return:
    """
    return

if __name__ == "__main__":
    usage="usage: %prog [options] FileName.fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--outdir",dest='out_dir', default='./',
                      help="Directory for output images default: ./")
    parser.add_option('--factor',dest='factor',type='int',nargs=2, default=(2,2),
                      help='TODO')
    parser.add_option('--cores',dest='cores',type='int',
                      help='Number of cores to use. Default = all available.')
    (options, args) = parser.parse_args()

    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    #logging.info("This is BANE {0}-({1})".format(__version__,__date__))

    if len(args)<1:
        parser.print_help()
        sys.exit()
    else:
        filename = args[0]
    if not os.path.exists(filename):
        logging.error("File not found: {0} ".format(filename))
        sys.exit(1)

    section(filename,options.factor)