#! /usr/bin/env python
from __future__ import print_function

import os
import datetime
import healpy as hp #dev on 1.8.1
import numpy as np #dev on 1.8.1
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits


class Region(object):
    """
    A Region object represents a footprint on the sky. This is done in a way similar to a MOC.
    The region is stored as a list of healpix pixels, allowing for binary set-like operations.
    """

    def __init__(self, maxdepth=11):
        self.maxdepth = maxdepth
        self.pixeldict = dict((i, set()) for i in range(1, maxdepth+1))
        self.demoted = set()
        return

    def __repr__(self):
        return "Region with maximum depth {0}, and total area {1:5.2g} deg^2".format(self.maxdepth, self.get_area())

    def add_circles(self, ra_cen, dec_cen, radius, depth=None):
        """
        Add one or more circles to this region
        :param ra_cen: ra or list of ras for circle centers
        :param dec_cen: dec or list of decs for circle centers
        :param radius: radius or list of radii for circles
        :param depth: The depth at which we wish to represent the circle (forced to be <=maxdepth)
        :return: None
        """
        if depth is None or depth > self.maxdepth:
            depth = self.maxdepth
        try:
            sky = list(zip(ra_cen, dec_cen))
            rad = radius
        except TypeError:
            sky = [[ra_cen, dec_cen]]
            rad = [radius]
        vectors = self.sky2vec(sky)
        for vec, r in zip(vectors, rad):
            pix = hp.query_disc(2**depth, vec, r, inclusive=True, nest=True)
            self.add_pixels(pix, depth)
        self._renorm()
        return

    def add_poly(self, positions, depth=None):
        """
        Add a single polygon to this region
        :param positions: list of [ (ra,dec), ... ] positions that form the polygon
        :param depth: The depth at which we wish to represent the circle (forced to be <=maxdepth
        :return: None
        """
        assert len(positions) >= 3, "A minimum of three coordinate pairs are required"

        if depth is None or depth > self.maxdepth:
            depth = self.maxdepth

        ras, decs = list(zip(*positions))
        sky = self.radec2sky(ras, decs)
        pix = hp.query_polygon(2**depth, self.sky2vec(sky), inclusive=True, nest=True)
        self.add_pixels(pix, depth)
        self._renorm()
        return

    def add_pixels(self, pix, depth):
        if depth not in self.pixeldict:
            self.pixeldict[depth] = set()
        self.pixeldict[depth].update(set(pix))
        pass

    def get_area(self, degrees=True):
        area = 0
        for d in range(1, self.maxdepth+1):
            area += len(self.pixeldict[d])*hp.nside2pixarea(2**d, degrees=degrees)
        return area

    def get_demoted(self):
        """
        :return: Return a set of pixels that represent this region at maxdepth
        """
        self._demote_all()
        return self.demoted

    def _demote_all(self):
        """
        Represent this region as pixels at maxdepth only
        """
        # only do the calculations if the demoted list is empty
        if len(self.demoted) == 0:
            pd = self.pixeldict
            for d in range(1, self.maxdepth):
                for p in pd[d]:
                    pd[d+1].update(set((4*p, 4*p+1, 4*p+2, 4*p+3)))
                pd[d] = set()  # clear the pixels from this level
            self.demoted = pd[d+1]
        return

    def _renorm(self):
        """
        Remake the pixel dictionary, merging groups of pixels at level N into a single pixel
        at level N-1
        """
        self.demoted = set()
        # convert all to lowest level
        self._demote_all()
        # now promote as needed
        for d in range(self.maxdepth, 2, -1):
            plist = self.pixeldict[d].copy()
            for p in plist:
                if p % 4 == 0:
                    nset = set((p, p+1, p+2, p+3))
                    if p+1 in plist and p+2 in plist and p+3 in plist:
                        # remove the four pixels from this level
                        self.pixeldict[d].difference_update(nset)
                        # add a new pixel to the next level up
                        self.pixeldict[d-1].add(p/4)
        self.demoted = set()
        return

    #@profile
    def sky_within(self, ra, dec, degin=False):
        """
        Test whether a sky position is within this region
        :param ra: RA in radians
        :param dec: Dec in radians
        :param degin: True if the input parameters are in degrees instead of radians
        :return: True if RA/Dec is within this region
        """
        sky = self.radec2sky(ra, dec)

        if degin:
            sky = np.radians(sky)

        theta_phi = self.sky2ang(sky)
        # cut out any entries that have nan for theta or phi
        mask = np.logical_and.reduce(np.isfinite(theta_phi), axis=0)
        theta_phi = theta_phi[:, mask]
        # need to check shape as the above can give 'empty' arrays of shape (22,0)
        if theta_phi.shape[1] < 1:
            return False

        theta, phi = theta_phi.transpose()
        pix = hp.ang2pix(2**self.maxdepth, theta, phi, nest=True)
        pixelset = self.get_demoted()
        result = np.in1d(pix, list(pixelset))
        return result

    def union(self, other, renorm=True):
        """
        Add another Region by performing union on their pixlists
        :param other: A Region
        """
        # merge the pixels that are common to both
        for d in range(1, min(self.maxdepth, other.maxdepth)+1):
            self.add_pixels(other.pixeldict[d], d)

        # if the other region is at higher resolution, then include a degraded version of the remaining pixels.
        if self.maxdepth < other.maxdepth:
            for d in range(self.maxdepth+1, other.maxdepth+1):
                for p in other.pixeldict[d]:
                    # promote this pixel to self.maxdepth
                    pp = p/4**(d-self.maxdepth)
                    self.pixeldict[self.maxdepth].add(pp)
        if renorm:
            self._renorm()
        return

    def without(self, other):
        """
        Remove the overlap between this region and the other region
        :param other: Another region
        :return: None
        """
        # work only on the lowest level
        # TODO: Allow this to be done for regions with different depths.
        assert self.maxdepth == other.maxdepth, "Regions must have the same maxdepth"
        self._demote_all()
        opd = set(other.get_demoted())
        self.pixeldict[self.maxdepth].difference_update(opd)
        self._renorm()
        return

    def intersect(self, other):
        """
        intersect this region with another
        :param other: a region
        :return: None
        """
        # work only on the lowest level
        # TODO: Allow this to be done for regions with different depths.
        assert self.maxdepth == other.maxdepth, "Regions must have the same maxdepth"
        self._demote_all()
        opd = set(other.get_demoted())
        self.pixeldict[self.maxdepth].intersection_update(opd)
        self._renorm()
        return

    def symmetric_difference(self, other):
        """

        :param other:
        :return:
        """
        # work only on the lowest level
        # TODO: Allow this to be done for regions with different depths.
        assert self.maxdepth == other.maxdepth, "Regions must have the same maxdepth"
        self._demote_all()
        opd = set(other.get_demoted())
        self.pixeldict[self.maxdepth].symmetric_difference_update(opd)
        self._renorm()
        return

    def write_reg(self, filename):
        """
        Write a ds9 region file that represents this region as a set of diamonds.
        :param filename: file to write
        :return: None
        """
        with open(filename, 'w') as out:
            for d in range(1, self.maxdepth+1):
                for p in self.pixeldict[d]:
                    line = "fk5; polygon("
                    # the following int() gets around some problems with np.int64 that exist prior to numpy v 1.8.1
                    vectors = list(zip(*hp.boundaries(2**d, int(p), step=1, nest=True)))
                    positions = []
                    for sky in self.vec2sky(np.array(vectors), degrees=True):
                        ra, dec = sky
                        pos = SkyCoord(ra/15, dec, unit=(u.degree, u.degree))
                        positions.append(pos.ra.to_string(sep=':', precision=2))
                        positions.append(pos.dec.to_string(sep=':', precision=2))
                    line += ','.join(positions)
                    line += ")"
                    print(line, file=out)
        return

    def write_fits(self, filename, moctool=''):
        """
        Write a fits file representing the MOC of this region.
        :param filename: Output filename
        :return: None
        """
        datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'MOC.fits')
        hdulist = fits.open(datafile)
        cols = fits.Column(name='NPIX', array=self._uniq(), format='1K')
        tbhdu = fits.BinTableHDU.from_columns([cols])
        hdulist[1] = tbhdu
        hdulist[1].header['PIXTYPE'] = ('HEALPIX ', 'HEALPix magic code')
        hdulist[1].header['ORDERING'] = ('NUNIQ ', 'NUNIQ coding method')
        hdulist[1].header['COORDSYS'] = ('C ', 'ICRS reference frame')
        hdulist[1].header['MOCORDER'] = (self.maxdepth, 'MOC resolution (best order)')
        hdulist[1].header['MOCTOOL'] = (moctool, 'Name of the MOC generator')
        hdulist[1].header['MOCTYPE'] = ('CATALOG', 'Source type (IMAGE or CATALOG)')
        hdulist[1].header['MOCID'] = (' ', 'Identifier of the collection')
        hdulist[1].header['ORIGIN'] = (' ', 'MOC origin')
        time = datetime.datetime.utcnow()
        hdulist[1].header['DATE'] = (datetime.datetime.strftime(time, format="%Y-%m-%dT%H:%m:%SZ"), 'MOC creation date')
        hdulist.writeto(filename, clobber=True)
        return

    def _uniq(self):
        """
        Create a list of all the pixels that cover this region.
        This list contains overlapping pixels of different orders.
        :return: A list of HealPix pixel numbers.
        """
        pd = []
        for d in range(1, self.maxdepth):
            pd.extend(map(lambda x: int(4**(d+1) + x), self.pixeldict[d]))
        return sorted(pd)

    @staticmethod
    def radec2sky(ra, dec):
        """
        Convert [ra], [dec] to [(ra[0], dec[0]),....]
        and also  ra,dec to [(ra,dec)] if ra/dec are not iterable
        :param ra: float or list of floats
        :param dec: float or list of floats
        :return: list of (ra,dec) tuples
        """
        try:
            sky = list(zip(ra,dec))
            #sky = np.empty((len(ra), 2), dtype=type(ra[0]))
            #sky[:, 0] = ra
            #sky[:, 1] = dec
        except TypeError:
            sky= [(ra,dec)]
        return sky

    @staticmethod
    def sky2ang(sky):
        """
        Convert ra,dec coordinates to theta,phi coordinates
        ra -> phi
        dec -> theta
        :param sky: float [(ra,dec),...]
        :return: A list of [(theta,phi), ...]
        """
        try:
            theta_phi = sky.copy()
        except AttributeError as e:
            theta_phi = np.array(sky)
        theta_phi[:, [1, 0]] = theta_phi[:, [0, 1]]
        theta_phi[:, 0] = np.pi/2 - theta_phi[:, 0]
        # # force 0<=theta<=2pi
        # theta_phi[:, 0] -= 2*np.pi*(theta_phi[:, 0]//(2*np.pi))
        # # and now -pi<=theta<=pi
        # theta_phi[:, 0] -= (theta_phi[:, 0] > np.pi)*2*np.pi
        return theta_phi

    @classmethod
    def sky2vec(cls, sky):
        """
        Convert sky positions in to 3d-vectors
        :param sky: [(ra,dec), ...]
        :return: [(x,y,z), ...]
        """
        theta_phi = cls.sky2ang(sky)
        theta, phi = map(np.array, list(zip(*theta_phi)))
        vec = hp.ang2vec(theta, phi)
        return vec

    @classmethod
    def vec2sky(cls, vec, degrees=False):
        """
        Convert [x,y,z] vectors into sky coordinates ra,dec
        :param vec: An array-like list of ([x,y,z],...)
        :param degrees: Return ra/dec in degrees? Default = false
        :return: [(ra,...),(dec,...)]
        """
        theta, phi = hp.vec2ang(vec)
        ra = phi
        dec = np.pi/2-theta

        if degrees:
            ra = np.degrees(ra)
            dec = np.degrees(dec)
        return cls.radec2sky(ra, dec)