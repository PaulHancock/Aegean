#! /usr/bin/env python

"""
MIMAS - The Multi-resolution Image Mask for Aegean Software

Created: Paul Hancock, Oct 2014
"""

import argparse
import logging
import numpy as np
import sys
import os
from astropy.io import fits as pyfits
from astropy.wcs import wcs as pywcs
from AegeanTools.regions import Region

version='v1.0'

#seems like this fails sometimes
try:
    import cPickle as pickle
except ImportError:
    import pickle

#globals
filewcs=None

#@profile
def maskfile(regionfile,infile,outfile):
    """
    Created a masked version of file, using region.
    This does not change the shape or size of the image, it just sets some pixels to be null/nan
    :param regionfile: A Region that describes the area of interest
    :param infile: The name of the fits file to mask.
    :param outfile: The masked file to be written
    :return: None
    """
    #Check that the input file is accessible and then open it
    assert os.path.exists(infile), "Cannot locate fits file {0}".format(infile)
    im = pyfits.open(infile)
    assert os.path.exists(regionfile), "Cannot locate region file {0}".format(regionfile)
    region=pickle.load(open(regionfile))
    #fix possible problems with miriad generated fits files % HT John Morgan.
    try:
        wcs = pywcs.WCS(im[0].header, naxis=2)
    except:
        wcs = pywcs.WCS(str(im[0].header),naxis=2)
    data = np.squeeze(im[0].data)

    #easy/slow version
    #TODO: revise this to be faster if at all possible.
    print data.shape
    for i,row in enumerate(data):
        skybox = wcs.wcs_pix2world([[i,j] for j in xrange(len(row))],1)
        ra,dec = zip(*skybox)
        mask = [ not a for a in region.sky_within(ra, dec, degin=True)]
        data[mask]=np.nan

    im[0].data=data
    im.writeto(outfile,clobber=True)
    logging.info("Wrote {0}".format(outfile))
    return

def mim2reg(mimfile,regfile):
    region=pickle.load(open(mimfile,'r'))
    region.write_reg(regfile)
    logging.info("Converted {0} -> {1}".format(mimfile,regfile))
    return

def combine_regions(container):
    """
    Return a region that is the combination of those specified in the container.
    The container is typically a results instance that comes from argparse.
    Any object with the properties [maxdepth,add_region,rem_region,include_circles, exclude_circles] will work
    :param container: Object containing the region descriptions.
    :return: A region
    """
    #create empty region
    region=Region(container.maxdepth)

    #add/rem all the regions from files
    for r in container.add_region:
        logging.info("adding region from {0}".format(r))
        r2=pickle.load(r)
        region.union(r2)

    for r in container.rem_region:
        logging.info("removing region from {0}".format(r))
        r2=pickle.load(r)
        region.without(r2)


    #add circles
    if len(container.include_circles) > 0:
        circles = np.radians(container.include_circles)
        ras,decs,radii=zip(*circles)
        region.add_circles(ras, decs, radii)

    #remove circles
    if len(container.exclude_circles) > 0:
        r2=Region(container.maxdepth)
        circles = np.radians(container.include_circles)
        ras,decs,radii=zip(*circles)
        r2.add_circles(ras, decs, radii)
        region.without(r2)

    #add polygons
    if len(container.include_polygons) > 0:
        poly = np.array(np.radians(container.include_polygons))
        poly = poly.reshape((poly.shape[0]/2,2))
        region.add_poly(poly)

    #remove polygons
    if len(container.exclude_polygons) > 0:
        poly = np.radians(container.include_polygons)
        r2 = Regions(container.maxdepth)
        r2.add_poly(poly)
        region.without(r2)

    return region

def save_region(region,filename):
    """
    Save the given region to a file
    :param region: A Region
    :param filename: A Filename
    :return: None
    """
    pickle.dump(region,open(filename,'w'),protocol=-1)
    logging.info("Wrote {0}".format(filename))
    return

if __name__=="__main__":
    epilog='Regions are added/subtracted in the following order, +r -r +c -c +p -p. This means that you might have to take multiple passes to construct overly complicated regions.'
    parser = argparse.ArgumentParser(epilog=epilog,prefix_chars='+-')

    group1=parser.add_argument_group('Creating/modifying regions','Must specify -o, plus or more [+-][cr]')
    #tools for creating .mim files
    group1.add_argument('-o', dest='outfile', action='store', help='output filename', default=None)
    group1.add_argument('-depth', dest='maxdepth',action='store',
                        metavar='N', default=8, type=int,
                        help='maximum nside=2**N to be used to represent this region. [Default=8]')
    group1.add_argument('+r', dest='add_region', action='append',
                        default=[], type=str, metavar='filename',nargs='*',
                        help='add a region specified by the given file (.mim format)')
    group1.add_argument('-r', dest='rem_region', action='append',
                        default=[], type=str, metavar='filename',nargs='*',
                        help='exclude a region specified by the given file ( .mim format)')
    #add/remove circles
    group1.add_argument('+c', dest='include_circles', action='append',
                        default=[], type=float, metavar=('ra','dec','radius'), nargs=3,
                        help='add a circle to this region (decimal degrees)')
    group1.add_argument('-c', dest='exclude_circles', action='append',
                        default=[],type=float, metavar=('ra','dec','radius'), nargs=3,
                        help='exclude the given circles from a region')

    #add/remove polygons
    group1.add_argument('+p', dest='include_polygons', action='store',
                        default=[], type=float, metavar=('ra','dec'), nargs='*',
                        help='add a polygon to this region ( decimal degrees)')
    group1.add_argument('-p', dest='exclude_polygons', action='store',
                        default=[], type=float, metavar=('ra','dec'), nargs='*',
                        help='remove a polygon from this region ( decimal degrees)')

    group2 = parser.add_argument_group("Using already created regions")
    #tools that use .mim files
    group2.add_argument('--mim2reg',dest='mim2reg', action='append',
                        type=str, metavar=('region.mim','region.reg'), nargs=2,
                        help='convert region.mim into region.reg', default=[])
    group2.add_argument('--mask',dest='mask', action='store',
                        type=str, metavar=('region.mim','file.fits','masked.fits'), nargs=3, default=[],
                        help='use region.mim to mask file.fits and write masekd.fits')
    group3 = parser.add_argument_group('Extra options')
    #extras
    group3.add_argument('--debug', dest='debug', action='store_true', help='debug mode [default=False]', default=False)
    group3.add_argument('--version', action='version', version='%(prog)s '+version)
    results = parser.parse_args()

    #TODO: see if there is an 'argparse' way of detecting no input
    if len(sys.argv)<=1:
        parser.print_help()
        sys.exit()

    logging_level = logging.DEBUG if results.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is MIMAS {0}".format(version))

    if len(results.mim2reg)>0:
        for i,o in results.mim2reg:
            mim2reg(i,o)
        sys.exit()

    if len(results.mask)>0:
        m, i, o = results.mask
        maskfile(m, i, o)
        sys.exit()

    if results.outfile is not None:
        region=combine_regions(results)
        save_region(region,results.outfile)
