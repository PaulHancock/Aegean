#! /usr/bin/env python

"""
MIMAS - The Multi-resolution Image Mask for Aegean Software

Created: Paul Hancock, Oct 2014

TODO: Write an in/out reader for MOC formats described by
http://arxiv.org/abs/1505.02937

"""

import argparse
import logging
import numpy as np
import sys
import os
import re
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.io import fits as pyfits
from astropy.wcs import wcs as pywcs
from AegeanTools.regions import Region

__version__ = 'v1.2'
__date__ = '2015-10-20'

# seems like this fails sometimes
try:
    import cPickle as pickle
except ImportError:
    import pickle

# globals
filewcs = None


class Dummy():
    def __init__(self, maxdepth=8):
        self.add_region = []
        self.rem_region = []
        self.include_circles = []
        self.exclude_circles = []
        self.include_polygons = []
        self.exclude_polygons = []
        self.maxdepth = maxdepth
        self.galactic = False
        return


def galactic2fk5(l, b):
    """
    Convert galactic l/b to fk5 ra/dec
    :param l: longitude in radians
    :param b: latitude in radians
    :return: ra, dec in radians
    """
    a = SkyCoord(l, b, unit=(u.radian, u.radian), frame='galactic')
    return a.fk5.ra.radian, a.fk5.dec.radian


def mask_plane(data, wcs, region, negate=False):
    """
    Mask a 2d image (data) such that pixels within 'region' are set to nan.
    :param data: 2d numpy.array
    :param wcs: a WCS object
    :param region: a MIMAS region
    :param negate: If True then pixels *outside* the region are set to nan.
    :return: the masked data (which is modified in place anyway)
    """
    # create an array but don't set the values (they are random)
    indexes = np.empty((data.shape[0]*data.shape[1], 2), dtype=int)
    # since I know exactly what the index array needs to look like i can construct
    # it faster than list comprehension would allow
    # we do this only once and then recycle it
    idx = np.array([(j, 0) for j in xrange(data.shape[1])])
    j = data.shape[1]
    for i in xrange(data.shape[0]):
        idx[:, 1] = i
        indexes[i*j:(i+1)*j] = idx

    # put ALL the pixles into our vectorized functions and minimise our overheads
    ra, dec = wcs.wcs_pix2world(indexes, 1).transpose()
    bigmask = region.sky_within(ra, dec, degin=True)
    if not negate:
        bigmask = np.bitwise_not(bigmask)
    # rework our 1d list into a 2d array
    bigmask = bigmask.reshape(data.shape)
    # and apply the mask
    data[bigmask] = np.nan
    return data


def maskfile(regionfile, infile, outfile, negate=False):
    """
    Created a masked version of file, using region.
    This does not change the shape or size of the image, it just sets some pixels to be null/nan
    :param regionfile: A Region that describes the area of interest
    :param infile: The name of the fits file to mask.
    :param outfile: The masked file to be written
    :param negate: Keep pixles that are outside the supplied region
    :return: None
    """
    # Check that the input file is accessible and then open it
    assert os.path.exists(infile), "Cannot locate fits file {0}".format(infile)
    im = pyfits.open(infile)
    assert os.path.exists(regionfile), "Cannot locate region file {0}".format(regionfile)
    region = pickle.load(open(regionfile))
    try:
        wcs = pywcs.WCS(im[0].header, naxis=2)
    except:
        wcs = pywcs.WCS(str(im[0].header), naxis=2)

    if len(im[0].data.shape) > 2:
        data = np.squeeze(im[0].data)
    else:
        data = im[0].data

    print data.shape
    if len(data.shape) == 3:
        for plane in range(data.shape[0]):
            mask_plane(data[plane], wcs, region, negate)
    else:
        mask_plane(data, wcs, region, negate)
    im[0].data = data
    im.writeto(outfile, clobber=True)
    logging.info("Wrote {0}".format(outfile))
    return


def mim2reg(mimfile, regfile):
    region = pickle.load(open(mimfile, 'r'))
    region.write_reg(regfile)
    logging.info("Converted {0} -> {1}".format(mimfile, regfile))
    return


def mim2fits(mimfile, fitsfile):
    region = pickle.load(open(mimfile, 'r'))
    region.write_fits(fitsfile, moctool='MIMAS {0}-{1}'.format(__version__, __date__))
    logging.info("Converted {0} -> {1}".format(mimfile, fitsfile))
    return


def box2poly(line):
    """
    Convert a line that describes a box in ds9 format, into a polygon that is given by the corners of the box
    :param line: text
    :return: list of [ ra,dec,ra,dec, ...  ]
    """
    words = re.split('[(\s,)]', line)
    ra = words[1]
    dec = words[2]
    width = words[3]
    height = words[4]
    if ":" in ra:
        ra = Angle(ra, unit=u.hour)
    else:
        ra = Angle(ra, unit=u.degree)
    dec = Angle(dec, unit=u.degree)
    width = Angle(float(width[:-1])/2, unit=u.arcsecond)  # strip the "
    height = Angle(float(height[:-1])/2, unit=u.arcsecond)  # strip the "
    center = SkyCoord(ra, dec)
    tl = center.ra.degree+width.degree, center.dec.degree+height.degree
    tr = center.ra.degree-width.degree, center.dec.degree+height.degree
    bl = center.ra.degree+width.degree, center.dec.degree-height.degree
    br = center.ra.degree-width.degree, center.dec.degree-height.degree
    return np.ravel([tl, tr, br, bl]).tolist()


def circle2circle(line):
    words = re.split('[(,\s)]', line)
    ra = words[1]
    dec = words[2]
    radius = words[3][:-1]  # strip the "
    if ":" in ra:
        ra = Angle(ra, unit=u.hour)
    else:
        ra = Angle(ra, unit=u.degree)
    dec = Angle(dec, unit=u.degree)
    radius = Angle(radius, unit=u.arcsecond)
    return [ra.degree, dec.degree, radius.degree]


def poly2poly(line):
    """
    This function works but the resulting polygons break healpy.
    :param line:
    :return:
    """
    words = re.split('[(\s,)]', line)
    ras = np.array(words[1::2])
    decs = np.array(words[2::2])
    coords = []
    for ra, dec in zip(ras, decs):
        if ra.strip() == '' or dec.strip() == '':
            continue
        if ":" in ra:
            pos = SkyCoord(Angle(ra, unit=u.hour), Angle(dec, unit=u.degree))
        else:
            pos = SkyCoord(Angle(ra, unit=u.degree), Angle(dec, unit=u.degree))
        # only add this point if it is some distance from the previous one
        coords.extend([pos.ra.degree, pos.dec.degree])
    return coords


def reg2mim(regfile, mimfile, maxdepth):
    """
    Read a ds9 regions file and create a mim file from it
    :param regfile:
    :param mimfile:
    :return:
    """
    logging.info("Reading regions from {0}".format(regfile))
    lines = (l for l in open(regfile, 'r') if not l.startswith('#'))
    poly = []
    circles = []
    for line in lines:
        if line.startswith('box'):
            poly.append(box2poly(line))
        elif line.startswith('circle'):
            circles.append(circle2circle(line))
        elif line.startswith('polygon'):
            logging.warn("Polygons break a lot, but I'll try this one anyway.")
            poly.append(poly2poly(line))
        else:
            logging.warn("Not sure what to do with {0}".format(line[:-1]))
    container = Dummy(maxdepth=maxdepth)
    container.include_circles = circles
    container.include_polygons = poly

    region = combine_regions(container)
    save_region(region,mimfile)
    return


def combine_regions(container):
    """
    Return a region that is the combination of those specified in the container.
    The container is typically a results instance that comes from argparse.
    Any object with the properties [maxdepth,add_region,rem_region,include_circles, exclude_circles] will work
    :param container: Object containing the region descriptions.
    :return: A region
    """
    # create empty region
    region = Region(container.maxdepth)

    # add/rem all the regions from files
    for r in container.add_region:
        logging.info("adding region from {0}".format(r))
        r2 = pickle.load(open(r[0], 'r'))
        region.union(r2)

    for r in container.rem_region:
        logging.info("removing region from {0}".format(r))
        r2 = pickle.load(open(r[0], 'r'))
        region.without(r2)


    # add circles
    if len(container.include_circles) > 0:
        for c in container.include_circles:
            circles = np.radians(np.array(c))
            if container.galactic:
                l, b, radii = circles.reshape(3,circles.shape[0]/3)
                ras, decs = galactic2fk5(l,b)
            else:
                ras, decs, radii = circles.reshape(3, circles.shape[0]/3)
            region.add_circles(ras, decs, radii)

    # remove circles
    if len(container.exclude_circles) > 0:
        for c in container.exclude_circles:
            r2 = Region(container.maxdepth)
            circles = np.radians(np.array(c))
            if container.galactic:
                l, b, radii = circles.reshape(3,circles.shape[0]/3)
                ras, decs = galactic2fk5(l,b)
            else:
                ras, decs, radii = circles.reshape(3, circles.shape[0]/3)
            r2.add_circles(ras, decs, radii)
            region.without(r2)

    # add polygons
    if len(container.include_polygons) > 0:
        for p in container.include_polygons:
            poly = np.radians(np.array(p))
            poly = poly.reshape((poly.shape[0]/2, 2))
            region.add_poly(poly)

    # remove polygons
    if len(container.exclude_polygons) > 0:
        for p in container.include_polygons:
            poly = np.array(np.radians(p))
            r2 = Region(container.maxdepth)
            r2.add_poly(poly)
            region.without(r2)

    return region


def save_region(region, filename):
    """
    Save the given region to a file
    :param region: A Region
    :param filename: A Filename
    :return: None
    """
    pickle.dump(region, open(filename, 'w'), protocol=-1)
    logging.info("Wrote {0}".format(filename))
    return


def save_as_image(region, filename):
    """

    :param region:
    :param filename:
    :return:
    """
    import healpy as hp
    pixels = list(region.get_demoted())
    order = region.maxdepth
    m = np.arange(hp.nside2npix(2**order))
    m[:] = 0
    m[pixels] = 1
    hp.write_map(filename, m, nest=True, coord='C')
    return

if __name__ == "__main__":
    epilog = 'Regions are added/subtracted in the following order, +r -r +c -c +p -p. ' + \
             'This means that you might have to take multiple passes to construct overly complicated regions.'
    parser = argparse.ArgumentParser(epilog=epilog, prefix_chars='+-')

    group1 = parser.add_argument_group('Creating/modifying regions', 'Must specify -o, plus or more [+-][cr]')
    # tools for creating .mim files
    group1.add_argument('-o', dest='outfile', action='store', help='output filename', default=None)
    group1.add_argument('-depth', dest='maxdepth', action='store',
                        metavar='N', default=8, type=int,
                        help='maximum nside=2**N to be used to represent this region. [Default=8]')
    group1.add_argument('+r', dest='add_region', action='append',
                        default=[], type=str, metavar='filename', nargs='*',
                        help='add a region specified by the given file (.mim format)')
    group1.add_argument('-r', dest='rem_region', action='append',
                        default=[], type=str, metavar='filename', nargs='*',
                        help='exclude a region specified by the given file ( .mim format)')
    # add/remove circles
    group1.add_argument('+c', dest='include_circles', action='append',
                        default=[], type=float, metavar=('ra', 'dec', 'radius'), nargs=3,
                        help='add a circle to this region (decimal degrees)')
    group1.add_argument('-c', dest='exclude_circles', action='append',
                        default=[],type=float, metavar=('ra', 'dec', 'radius'), nargs=3,
                        help='exclude the given circles from a region')

    # add/remove polygons
    group1.add_argument('+p', dest='include_polygons', action='append',
                        default=[], type=float, metavar=('ra', 'dec'), nargs='*',
                        help='add a polygon to this region ( decimal degrees)')
    group1.add_argument('-p', dest='exclude_polygons', action='append',
                        default=[], type=float, metavar=('ra', 'dec'), nargs='*',
                        help='remove a polygon from this region ( decimal degrees)')
    group1.add_argument('-g', dest='galactic', action='store_true', default=False,
                        help='Interpret input coordinates are galactic instead of equatorial.')

    group2 = parser.add_argument_group("Using already created regions")
    # tools that use .mim files
    group2.add_argument('--mim2reg', dest='mim2reg', action='append',
                        type=str, metavar=('region.mim', 'region.reg'), nargs=2,
                        help='convert region.mim into region.reg', default=[])
    group2.add_argument('--reg2mim', dest='reg2mim', action='append',
                        type=str, metavar=('region.reg', 'region.mim'), nargs=2,
                        help="Converta a .reg file into a .mim file", default=[])
    group2.add_argument('--mim2fits',dest='mim2fits', action='append',
                        type=str, metavar=('region.mim', 'region_MOC.fits'), nargs=2,
                        help="Convert a .mim file into a MOC.fits file", default=[])
    group3 = parser.add_argument_group("Masking fits files")
    group3.add_argument('--mask', dest='mask', action='store',
                        type=str, metavar=('region.mim', 'file.fits', 'masked.fits'), nargs=3, default=[],
                        help='use region.mim to mask file.fits and write masekd.fits')
    group3.add_argument('--negate', dest='negate', action='store_true', default=False,
                        help='If True then mask pixels inside this region instead of outside.')
    group3.add_argument('--fitsmask', dest='fits_mask', action='store',
                        type=str, metavar=('mask.fits', 'file.fits','masked_file.fits'), nargs=3, default=[],
                        help='Use a fits file as a mask for another fits file. ' +
                             'Values of blank/nan/zero are considered to be mask=True.')
    group4 = parser.add_argument_group('Extra options')
    # extras
    group4.add_argument('--fitsimage', dest='mim2img', action='store_true',
                        default=False, help='Save the region as a fits image')
    group4.add_argument('--debug', dest='debug', action='store_true', help='debug mode [default=False]', default=False)
    group4.add_argument('--version', action='version', version='%(prog)s '+__version__+"-({0})".format(__date__))
    results = parser.parse_args()

    # TODO: see if there is an 'argparse' way of detecting no input
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    logging_level = logging.DEBUG if results.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is MIMAS {0}-({1})".format(__version__,__date__))

    if len(results.fits_mask) > 0:
        logging.info("The --fitsmask option is not yet implemented.")
        sys.exit(1)

    if len(results.mim2reg) > 0:
        for i, o in results.mim2reg:
            mim2reg(i, o)
        sys.exit()

    if len(results.reg2mim) > 0:
        for i, o in results.reg2mim:
            reg2mim(i, o, results.maxdepth)
        sys.exit()

    if len(results.mim2fits) > 0:
        for i, o in results.mim2fits:
            mim2fits(i, o)
        sys.exit()

    if len(results.mask) > 0:
        m, i, o = results.mask
        maskfile(m, i, o, results.negate)
        sys.exit()

    if results.mim2img:
        region = combine_regions(results)
        save_as_image(region, results.outfile)
        sys.exit()

    if results.outfile is not None:
        region = combine_regions(results)
        save_region(region, results.outfile)


