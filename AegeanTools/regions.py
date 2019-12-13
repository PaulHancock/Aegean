#! /usr/bin/env python
"""
Describe sky areas as a collection of HEALPix pixels
"""

from __future__ import print_function
import os
import datetime
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import six

if six.PY2:
    import cPickle
else:
    import _pickle as cPickle

__author__ = "Paul Hancock"


class Region(object):
    """
    A Region object represents a footprint on the sky. This is done in a way similar to a MOC.
    The region is stored as a list of healpix pixels, allowing for binary set-like operations.

    Attributes
    ----------
    maxdepth : int
        The depth or resolution of the region.
        At the deepest level there will be 4*2**maxdepth pixels on the sky.
        Default = 11

    pixeldict : dict
        A dictionary of sets, each set containing the pixels within the region. The sets are indexed by their
        layer number.

    demoted : set
        A representation of this region at the deepest layer.
    """

    def __init__(self, maxdepth=11):
        self.maxdepth = maxdepth
        self.pixeldict = dict((i, set()) for i in range(1, maxdepth+1))
        self.demoted = set()
        return

    @classmethod
    def load(cls, mimfile):
        """
        Create a region object from the given file.

        Parameters
        ----------
        mimfile : str
            File to load.

        Returns
        -------
        region : `AegeanTools.regions.Region`
            A region object
        """
        reg = cPickle.load(open(mimfile, 'rb'))
        return reg

    def save(self, mimfile):
        """
        Save this region to a file

        Parameters
        ----------
        mimfile : str
            File to write
        """
        cPickle.dump(self, open(mimfile,'wb'), protocol=2)
        return

    def __repr__(self):
        return "Region with maximum depth {0}, and total area {1:5.2g} deg^2".format(self.maxdepth, self.get_area())

    def add_circles(self, ra_cen, dec_cen, radius, depth=None):
        """
        Add one or more circles to this region

        Parameters
        ----------
        ra_cen, dec_cen, radius : float or list
            The center and radius of the circle or circles to add to this region.

        depth : int
            The depth at which the given circles will be inserted.

        """
        if depth is None or depth > self.maxdepth:
            depth = self.maxdepth
        try:
            sky = list(zip(ra_cen, dec_cen))
            rad = radius
        except TypeError:
            sky = [[ra_cen, dec_cen]]
            rad = [radius]
        sky = np.array(sky)
        rad = np.array(rad)
        vectors = self.sky2vec(sky)
        for vec, r in zip(vectors, rad):
            pix = hp.query_disc(2**depth, vec, r, inclusive=True, nest=True)
            self.add_pixels(pix, depth)
        self._renorm()
        return

    def add_poly(self, positions, depth=None):
        """
        Add a single polygon to this region.

        Parameters
        ----------
        positions : [[ra, dec], ...]
            Positions for the vertices of the polygon. The polygon needs to be convex and non-intersecting.

        depth : int
            The deepth at which the polygon will be inserted.
        """
        if not (len(positions) >= 3): raise AssertionError("A minimum of three coordinate pairs are required")

        if depth is None or depth > self.maxdepth:
            depth = self.maxdepth

        ras, decs = np.array(list(zip(*positions)))
        sky = self.radec2sky(ras, decs)
        pix = hp.query_polygon(2**depth, self.sky2vec(sky), inclusive=True, nest=True)
        self.add_pixels(pix, depth)
        self._renorm()
        return

    def add_pixels(self, pix, depth):
        """
        Add one or more HEALPix pixels to this region.

        Parameters
        ----------
        pix : int or iterable
            The pixels to be added

        depth : int
            The depth at which the pixels are added.
        """
        if depth not in self.pixeldict:
            self.pixeldict[depth] = set()
        self.pixeldict[depth].update(set(pix))

    def get_area(self, degrees=True):
        """
        Calculate the total area represented by this region.

        Parameters
        ----------
        degrees : bool
            If True then return the area in square degrees, otherwise use steradians.
            Default = True.

        Returns
        -------
        area : float
            The area of the region.
        """
        area = 0
        for d in range(1, self.maxdepth+1):
            area += len(self.pixeldict[d])*hp.nside2pixarea(2**d, degrees=degrees)
        return area

    def get_demoted(self):
        """
        Get a representation of this region at the deepest level.

        Returns
        -------
        demoted : set
            A set of pixels, at the highest resolution.
        """
        self._demote_all()
        return self.demoted

    def _demote_all(self):
        """
        Convert the multi-depth pixeldict into a single set of pixels at the deepest layer.

        The result is cached, and reset when any changes are made to this region.
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

    def sky_within(self, ra, dec, degin=False):
        """
        Test whether a sky position is within this region

        Parameters
        ----------
        ra, dec : float
            Sky position.

        degin : bool
            If True the ra/dec is interpreted as degrees, otherwise as radians.
            Default = False.

        Returns
        -------
        within : bool
            True if the given position is within one of the region's pixels.
        """
        sky = self.radec2sky(ra, dec)

        if degin:
            sky = np.radians(sky)

        theta_phi = self.sky2ang(sky)
        # Set values that are nan to be zero and record a mask
        mask = np.bitwise_not(np.logical_and.reduce(np.isfinite(theta_phi), axis=1))
        theta_phi[mask, :] = 0

        theta, phi = theta_phi.transpose()
        pix = hp.ang2pix(2**self.maxdepth, theta, phi, nest=True)
        pixelset = self.get_demoted()
        result = np.in1d(pix, list(pixelset))
        # apply the mask and set the shonky values to False
        result[mask] = False
        return result

    def union(self, other, renorm=True):
        """
        Add another Region by performing union on their pixlists.

        Parameters
        ----------
        other : :class:`AegeanTools.regions.Region`
            The region to be combined.

        renorm : bool
            Perform renormalisation after the operation?
            Default = True.
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
        Subtract another Region by performing a difference operation on their pixlists.

        Requires both regions to have the same maxdepth.

        Parameters
        ----------
        other : :class:`AegeanTools.regions.Region`
            The region to be combined.
        """
        # work only on the lowest level
        # TODO: Allow this to be done for regions with different depths.
        if not (self.maxdepth == other.maxdepth): raise AssertionError("Regions must have the same maxdepth")
        self._demote_all()
        opd = set(other.get_demoted())
        self.pixeldict[self.maxdepth].difference_update(opd)
        self._renorm()
        return

    def intersect(self, other):
        """
        Combine with another Region by performing intersection on their pixlists.

        Requires both regions to have the same maxdepth.

        Parameters
        ----------
        other : :class:`AegeanTools.regions.Region`
            The region to be combined.
        """
        # work only on the lowest level
        # TODO: Allow this to be done for regions with different depths.
        if not (self.maxdepth == other.maxdepth): raise AssertionError("Regions must have the same maxdepth")
        self._demote_all()
        opd = set(other.get_demoted())
        self.pixeldict[self.maxdepth].intersection_update(opd)
        self._renorm()
        return

    def symmetric_difference(self, other):
        """
        Combine with another Region by performing the symmetric difference of their pixlists.

        Requires both regions to have the same maxdepth.

        Parameters
        ----------
        other : :class:`AegeanTools.regions.Region`
            The region to be combined.
        """
        # work only on the lowest level
        # TODO: Allow this to be done for regions with different depths.
        if not (self.maxdepth == other.maxdepth): raise AssertionError("Regions must have the same maxdepth")
        self._demote_all()
        opd = set(other.get_demoted())
        self.pixeldict[self.maxdepth].symmetric_difference_update(opd)
        self._renorm()
        return

    def write_reg(self, filename):
        """
        Write a ds9 region file that represents this region as a set of diamonds.

        Parameters
        ----------
        filename : str
            File to write
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

        Parameters
        ----------
        filename : str
            File to write

        moctool : str
            String to be written to fits header with key "MOCTOOL".
            Default = ''
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
        hdulist.writeto(filename, overwrite=True)
        return

    def _uniq(self):
        """
        Create a list of all the pixels that cover this region.
        This list contains overlapping pixels of different orders.

        Returns
        -------
        pix : list
            A list of HEALPix pixel numbers.
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

        Parameters
        ----------
        ra, dec : float or iterable
            Sky coordinates

        Returns
        -------
        sky : numpy.array
            array of (ra,dec) coordinates.
        """
        try:
            sky = np.array(list(zip(ra, dec)))
        except TypeError:
            sky = np.array([(ra, dec)])
        return sky

    @staticmethod
    def sky2ang(sky):
        """
        Convert ra,dec coordinates to theta,phi coordinates
        ra -> phi
        dec -> theta

        Parameters
        ----------
        sky : numpy.array
            Array of (ra,dec) coordinates.
            See :func:`AegeanTools.regions.Region.radec2sky`

        Returns
        -------
        theta_phi : numpy.array
            Array of (theta,phi) coordinates.
        """
        try:
            theta_phi = sky.copy()
        except AttributeError as _:
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
        Convert sky positions in to 3d-vectors on the unit sphere.

        Parameters
        ----------
        sky : numpy.array
            Sky coordinates as an array of (ra,dec)

        Returns
        -------
        vec : numpy.array
            Unit vectors as an array of (x,y,z)

        See Also
        --------
        :func:`AegeanTools.regions.Region.vec2sky`
        """
        theta_phi = cls.sky2ang(sky)
        theta, phi = map(np.array, list(zip(*theta_phi)))
        vec = hp.ang2vec(theta, phi)
        return vec

    @classmethod
    def vec2sky(cls, vec, degrees=False):
        """
        Convert [x,y,z] vectors into sky coordinates ra,dec

        Parameters
        ----------
        vec : numpy.array
            Unit vectors as an array of (x,y,z)

        degrees

        Returns
        -------
        sky : numpy.array
            Sky coordinates as an array of (ra,dec)

        See Also
        --------
        :func:`AegeanTools.regions.Region.sky2vec`
        """
        theta, phi = hp.vec2ang(vec)
        ra = phi
        dec = np.pi/2-theta

        if degrees:
            ra = np.degrees(ra)
            dec = np.degrees(dec)
        return cls.radec2sky(ra, dec)
