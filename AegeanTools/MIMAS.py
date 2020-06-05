#! /usr/bin/env python
"""
MIMAS - The Multi-resolution Image Mask for Aegean Software

TODO: Write an in/out reader for MOC formats described by
http://arxiv.org/abs/1505.02937
"""

from __future__ import print_function

import logging
import numpy as np

import os
import re
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.io import fits as pyfits
from astropy.wcs import wcs as pywcs
import healpy as hp
from .regions import Region
from .catalogs import load_table, write_table

__author__ = "Paul Hancock"
__version__ = 'v1.3.1'
__date__ = '2018-08-29'


# globals
filewcs = None


class Dummy():
    """
    A state storage class for MIMAS to work with.

    Attributes
    ----------
    add_region : list
        List of :class:`AegeanTools.MIMAS.Region` to be added.

    rem_region : list
        List of :class:`AegeanTools.MIMAS.Region` to be subtracted.

    include_circles : [[ra, dec, radius],...]
        List of circles to be added to the region, units are degrees.

    exclude_circles : [[ra, dec, radius], ...]
        List of circles to be subtracted from the region, units are degrees.

    include_polygons : [[ra,dec, ...], ...]
        List of polygons to be added to the region, units are degrees.

    exclude_polygons : [[ra,dec, ...], ...]
        List of polygons to be subtracted from the region, units are degrees.

    maxdepth : int
        Depth or resolution of the region for HEALPix.
        There are 4*2**maxdepth pixels at the deepest layer.
        Default = 8.

    galactic: bool
        If true then all ra/dec coordinates will be interpreted as if they were in galactic
        lat/lon (degrees)
    """
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

    Parameters
    ----------
    l, b : float
        Galactic coordinates in radians.

    Returns
    -------
    ra, dec : float
        FK5 ecliptic coordinates in radians.
    """
    a = SkyCoord(l, b, unit=(u.radian, u.radian), frame='galactic')
    return a.fk5.ra.radian, a.fk5.dec.radian


def mask_plane(data, wcs, region, negate=False):
    """
    Mask a 2d image (data) such that pixels within 'region' are set to nan.

    Parameters
    ----------
    data : 2d-array
        Image array.

    wcs : astropy.wcs.WCS
        WCS for the image in question.

    region : :class:`AegeanTools.regions.Region`
        A region within which the image pixels will be masked.

    negate : bool
        If True then pixels *outside* the region are masked.
        Default = False.

    Returns
    -------
    masked : 2d-array
        The original array, but masked as required.
    """
    # create an array but don't set the values (they are random)
    indexes = np.empty((data.shape[0]*data.shape[1], 2), dtype=int)
    # since I know exactly what the index array needs to look like i can construct
    # it faster than list comprehension would allow
    # we do this only once and then recycle it
    idx = np.array([(j, 0) for j in range(data.shape[1])])
    j = data.shape[1]
    for i in range(data.shape[0]):
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


def mask_file(regionfile, infile, outfile, negate=False):
    """
    Created a masked version of file, using a region.


    Parameters
    ----------
    regionfile : str
        A file which can be loaded as a :class:`AegeanTools.regions.Region`.
        The image will be masked according to this region.

    infile : str
        Input FITS image.

    outfile : str
        Output FITS image.

    negate :  bool
        If True then pixels *outside* the region are masked.
        Default = False.

    See Also
    --------
    :func:`AegeanTools.MIMAS.mask_plane`
    """
    # Check that the input file is accessible and then open it
    if not os.path.exists(infile): raise AssertionError("Cannot locate fits file {0}".format(infile))
    im = pyfits.open(infile)
    if not os.path.exists(regionfile): raise AssertionError("Cannot locate region file {0}".format(regionfile))
    region = Region.load(regionfile)
    try:
        wcs = pywcs.WCS(im[0].header, naxis=2)
    except:  # TODO: figure out what error is being thrown
        wcs = pywcs.WCS(str(im[0].header), naxis=2)

    if len(im[0].data.shape) > 2:
        data = np.squeeze(im[0].data)
    else:
        data = im[0].data

    print(data.shape)
    if len(data.shape) == 3:
        for plane in range(data.shape[0]):
            mask_plane(data[plane], wcs, region, negate)
    else:
        mask_plane(data, wcs, region, negate)
    im[0].data = data
    im.writeto(outfile, overwrite=True)
    logging.info("Wrote {0}".format(outfile))
    return


def mask_table(region, table, negate=False, racol='ra', deccol='dec'):
    """
    Apply a given mask (region) to the table, removing all the rows with ra/dec inside the region
    If negate=False then remove the rows with ra/dec outside the region.


    Parameters
    ----------
    region : :class:`AegeanTools.regions.Region`
        Region to mask.

    table : Astropy.table.Table
        Table to be masked.

    negate :  bool
        If True then pixels *outside* the region are masked.
        Default = False.

    racol, deccol : str
        The name of the columns in `table` that should be interpreted as ra and dec.
        Default = 'ra', 'dec'

    Returns
    -------
    masked : Astropy.table.Table
        A view of the given table which has been masked.
    """
    inside = region.sky_within(table[racol], table[deccol], degin=True)
    if not negate:
        mask = np.bitwise_not(inside)
    else:
        mask = inside
    return table[mask]


def mask_catalog(regionfile, infile, outfile, negate=False, racol='ra', deccol='dec'):
    """
    Apply a region file as a mask to a catalog, removing all the rows with ra/dec inside the region
    If negate=False then remove the rows with ra/dec outside the region.


    Parameters
    ----------
    regionfile : str
        A file which can be loaded as a :class:`AegeanTools.regions.Region`.
        The catalogue will be masked according to this region.

    infile : str
        Input catalogue.

    outfile : str
        Output catalogue.

    negate :  bool
        If True then pixels *outside* the region are masked.
        Default = False.

    racol, deccol : str
        The name of the columns in `table` that should be interpreted as ra and dec.
        Default = 'ra', 'dec'

    See Also
    --------
    :func:`AegeanTools.MIMAS.mask_table`

    :func:`AegeanTools.catalogs.load_table`
    """
    logging.info("Loading region from {0}".format(regionfile))
    region = Region.load(regionfile)
    logging.info("Loading catalog from {0}".format(infile))
    table = load_table(infile)
    masked_table = mask_table(region, table, negate=negate, racol=racol, deccol=deccol)
    write_table(masked_table, outfile)
    return


def mim2reg(mimfile, regfile):
    """
    Convert a MIMAS region (.mim) file into a DS9 region (.reg) file.

    Parameters
    ----------
    mimfile : str
        Input file in MIMAS format.

    regfile : str
        Output file.

    """
    region = Region.load(mimfile)
    region.write_reg(regfile)
    logging.info("Converted {0} -> {1}".format(mimfile, regfile))
    return


def mim2fits(mimfile, fitsfile):
    """
    Convert a MIMAS region (.mim) file into a MOC region (.fits) file.

    Parameters
    ----------
    mimfile : str
        Input file in MIMAS format.

    fitsfile : str
        Output file.
    """
    region = Region.load(mimfile)
    region.write_fits(fitsfile, moctool='MIMAS {0}-{1}'.format(__version__, __date__))
    logging.info("Converted {0} -> {1}".format(mimfile, fitsfile))
    return


def mask2mim(maskfile, mimfile, threshold=1.0, maxdepth=8):
    """
    Use a fits file as a mask to create a region file.

    Pixels in mask file that are equal or above the threshold will be included in the reigon,
    while those that are below the threshold will not.

    Parameters
    ----------
    maskfile : str
        Input file in fits format.

    mimfile : str
        Output filename

    threshold : float
        threshold value for separating include/exclude values

    maxdepth : int
        Maximum depth (resolution) of the healpix pixels

    """
    hdu = pyfits.open(maskfile)
    wcs = pywcs.WCS(hdu[0].header)

    x, y = np.where(hdu[0].data >= threshold)
    ra, dec = wcs.all_pix2world(y, x, 0)
    sky = np.radians(Region.radec2sky(ra, dec))
    vec = Region.sky2vec(sky)
    x, y, z = np.transpose(vec)
    pix = hp.vec2pix(2**maxdepth, x, y, z, nest=True)

    region = Region(maxdepth=maxdepth)
    region.add_pixels(pix, depth=maxdepth)
    region._renorm()
    save_region(region, mimfile)
    logging.info("Converted {0} -> {1}".format(maskfile, mimfile))
    return


def box2poly(line):
    """
    Convert a string that describes a box in ds9 format, into a polygon that is given by the corners of the box

    Parameters
    ----------
    line : str
        A string containing a DS9 region command for a box.

    Returns
    -------
    poly : [ra, dec, ...]
        The corners of the box in clockwise order from top left.
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
    """
    Parse a string that describes a circle in ds9 format.

    Parameters
    ----------
    line : str
        A string containing a DS9 region command for a circle.

    Returns
    -------
    circle : [ra, dec, radius]
        The center and radius of the circle.
    """
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
    Parse a string of text containing a DS9 description of a polygon.

    This function works but is not very robust due to the constraints of healpy.

    Parameters
    ----------
    line : str
        A string containing a DS9 region command for a polygon.

    Returns
    -------
    poly : [ra, dec, ...]
        The coordinates of the polygon.
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
    Parse a DS9 region file and write a MIMAS region (.mim) file.

    Parameters
    ----------
    regfile : str
        DS9 region (.reg) file.

    mimfile : str
        MIMAS region (.mim) file.

    maxdepth : str
        Depth/resolution of the region file.

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
            logging.warning("Polygons break a lot, but I'll try this one anyway.")
            poly.append(poly2poly(line))
        else:
            logging.warning("Not sure what to do with {0}".format(line[:-1]))
    container = Dummy(maxdepth=maxdepth)
    container.include_circles = circles
    container.include_polygons = poly

    region = combine_regions(container)
    save_region(region, mimfile)
    return


def combine_regions(container):
    """
    Return a region that is the combination of those specified in the container.
    The container is typically a results instance that comes from argparse.

    Order of construction is: add regions, subtract regions, add circles, subtract circles,
    add polygons, subtract polygons.

    Parameters
    ----------
    container : :class:`AegeanTools.MIMAS.Dummy`
        The regions to be combined.

    Returns
    -------
    region : :class:`AegeanTools.regions.Region`
        The constructed region.
    """
    # create empty region
    region = Region(container.maxdepth)

    # add/rem all the regions from files
    for r in container.add_region:
        logging.info("adding region from {0}".format(r))
        r2 = Region.load(r[0])
        region.union(r2)

    for r in container.rem_region:
        logging.info("removing region from {0}".format(r))
        r2 = Region.load(r[0])
        region.without(r2)


    # add circles
    if len(container.include_circles) > 0:
        for c in container.include_circles:
            circles = np.radians(np.array(c))
            if container.galactic:
                l, b, radii = circles.reshape(3, circles.shape[0]//3)
                ras, decs = galactic2fk5(l, b)
            else:
                ras, decs, radii = circles.reshape(3, circles.shape[0]//3)
            region.add_circles(ras, decs, radii)

    # remove circles
    if len(container.exclude_circles) > 0:
        for c in container.exclude_circles:
            r2 = Region(container.maxdepth)
            circles = np.radians(np.array(c))
            if container.galactic:
                l, b, radii = circles.reshape(3, circles.shape[0]//3)
                ras, decs = galactic2fk5(l, b)
            else:
                ras, decs, radii = circles.reshape(3, circles.shape[0]//3)
            r2.add_circles(ras, decs, radii)
            region.without(r2)

    # add polygons
    if len(container.include_polygons) > 0:
        for p in container.include_polygons:
            poly = np.radians(np.array(p))
            poly = poly.reshape((poly.shape[0]//2, 2))
            region.add_poly(poly)

    # remove polygons
    if len(container.exclude_polygons) > 0:
        for p in container.include_polygons:
            poly = np.array(np.radians(p))
            r2 = Region(container.maxdepth)
            r2.add_poly(poly)
            region.without(r2)

    return region


def intersect_regions(flist):
    """
    Construct a region which is the intersection of all regions described in the given
    list of file names.

    Parameters
    ----------
    flist : list
        A list of region filenames.

    Returns
    -------
    region : :class:`AegeanTools.regions.Region`
        The intersection of all regions, possibly empty.
    """
    if len(flist) < 2:
        raise Exception("Require at least two regions to perform intersection")
    a = Region.load(flist[0])
    for b in [Region.load(f) for f in flist[1:]]:
        a.intersect(b)
    return a


def save_region(region, filename):
    """
    Save the given region to a file

    Parameters
    ----------
    region : :class:`AegeanTools.regions.Region`
        A region.

    filename : str
        Output file name.
    """
    region.save(filename)
    logging.info("Wrote {0}".format(filename))
    return


def save_as_image(region, filename):
    """
    Convert a MIMAS region (.mim) file into a image (eg .png)

    Parameters
    ----------
    region : :class:`AegeanTools.regions.Region`
        Region of interest.

    filename : str
        Output filename.
    """
    import healpy as hp
    pixels = list(region.get_demoted())
    order = region.maxdepth
    m = np.arange(hp.nside2npix(2**order))
    m[:] = 0
    m[pixels] = 1
    hp.write_map(filename, m, nest=True, coord='C')
    return
