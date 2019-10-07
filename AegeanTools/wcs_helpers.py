#! /usr/bin/env python
"""
This module contains two classes that provide WCS functions that are not
part of the WCS toolkit, as well as some wrappers around the provided tools
to make them a lot easier to use.
"""

from __future__ import print_function

import astropy.wcs as pywcs
from astropy.io import fits
import numpy as np

# AegeanTools
from .angle_tools import gcd, bear, translate

# join the Aegean logger
import logging

__author__ = 'Paul Hancock'

log = logging.getLogger('Aegean')


class WCSHelper(object):
    """
    A wrapper around astropy.wcs that provides extra functionality, and hides the c/fortran indexing troubles.

    Additionally allow psf information to be described in a map instead of the fits header of the image.

    Useful functions not provided by astropy.wcs

    - sky2pix/pix2sky functions for vectors and ellipses.
    - functions for calculating the beam in sky/pixel coords
    - the ability to change the beam according to dec-lat
    """

    def __init__(self, wcs, beam, pixscale, refpix, lat=None, psf=None):
        """
        Parameters
        ----------
        wcs : :class:`astropy.wcs.WCS`
            WCS object

        beam : :class:`AegeanTools.wcs_helpers.Beam`
            The synthesized beam.

        pixscale : (float, float)
            The pixel scale at the reference location (degrees)

        refpix : (float, float)
            The reference location in pixel coordinates

        lat : float
            The latitude of the telescope

        psf : str
            Filename for a psf map
        """
        self.wcs = wcs
        self.beam = beam
        self.pixscale = pixscale
        self.refpix = refpix
        self.lat = lat

    @classmethod
    def from_header(cls, header, beam=None, lat=None):
        """
        Create a new WCSHelper class from the given header.

        Parameters
        ----------
        header : `astropy.fits.HDUHeader` or string
            The header to be used to create the WCS helper

        beam : :class:`AegeanTools.wcs_helpers.Beam` or None
            The synthesized beam. If the supplied beam is None then one is constructed form the header.

        lat : float
            The latitude of the telescope.

        Returns
        -------
        obj : :class:`AegeanTools.wcs_helpers.WCSHelper`
            A helper object.
        """
        try:
            wcs = pywcs.WCS(header, naxis=2)
        except:  # TODO: figure out what error is being thrown
            wcs = pywcs.WCS(str(header), naxis=2)

        if beam is None:
            beam = get_beam(header)
        else:
            beam = beam

        if beam is None:
            logging.critical("Cannot determine beam information")

        _, pixscale = get_pixinfo(header)
        refpix = (header['CRPIX1'], header['CRPIX2'])
        return cls(wcs, beam, pixscale, refpix, lat)

    @classmethod
    def from_file(cls, filename, beam=None):
        """
        Create a new WCSHelper class from a given fits file.

        Parameters
        ----------
        filename : string
            The file to be read

        beam : :class:`AegeanTools.wcs_helpers.Beam` or None
            The synthesized beam. If the supplied beam is None then one is constructed form the header.

        Returns
        -------
        obj : :class:`AegeanTools.wcs_helpers.WCSHelper`
            A helper object
        """
        header = fits.getheader(filename)
        return cls.from_header(header, beam)

    def pix2sky(self, pixel):
        """
        Convert pixel coordinates into sky coordinates.

        Parameters
        ----------
        pixel : (float, float)
            The (x,y) pixel coordinates

        Returns
        -------
        sky : (float, float)
            The (ra,dec) sky coordinates in degrees

        """
        x, y = pixel
        # wcs and pyfits have oposite ideas of x/y
        return self.wcs.wcs_pix2world([[y, x]], 1)[0]

    def sky2pix(self, pos):
        """
        Convert sky coordinates into pixel coordinates.

        Parameters
        ----------
        pos : (float, float)
            The (ra, dec) sky coordinates (degrees)

        Returns
        -------
        pixel : (float, float)
            The (x,y) pixel coordinates

        """
        pixel = self.wcs.wcs_world2pix([pos], 1)
        # wcs and pyfits have oposite ideas of x/y
        return [pixel[0][1], pixel[0][0]]

    def sky2pix_vec(self, pos, r, pa):
        """
        Convert a vector from sky to pixel coords.
        The vector has a magnitude, angle, and an origin on the sky.

        Parameters
        ----------
        pos : (float, float)
            The (ra, dec) of the origin of the vector (degrees).

        r : float
            The magnitude or length of the vector (degrees).

        pa : float
            The position angle of the vector (degrees).

        Returns
        -------
        x, y : (float, float)
            The pixel coordinates of the origin.
        r, theta : float
            The magnitude (pixels) and angle (degrees) of the vector.

        """
        ra, dec = pos
        x, y = self.sky2pix(pos)
        a = translate(ra, dec, r, pa)
        locations = self.sky2pix(a)
        x_off, y_off = locations
        a = np.sqrt((x - x_off) ** 2 + (y - y_off) ** 2)
        theta = np.degrees(np.arctan2((y_off - y), (x_off - x)))
        return x, y, a, theta

    def pix2sky_vec(self, pixel, r, theta):
        """
        Given and input position and vector in pixel coordinates, calculate
        the equivalent position and vector in sky coordinates.

        Parameters
        ----------
        pixel : (float, float)
            origin of vector in pixel coordinates
        r : float
            magnitude of vector in pixels
        theta : float
            angle of vector in degrees

        Returns
        -------
        ra, dec : (float, float)
            The (ra, dec) of the origin point (degrees).
        r, pa : (float, float)
            The magnitude and position angle of the vector (degrees).
        """
        ra1, dec1 = self.pix2sky(pixel)
        x, y = pixel
        a = (x + r * np.cos(np.radians(theta)),
             y + r * np.sin(np.radians(theta)))
        locations = self.pix2sky(a)
        ra2, dec2 = locations
        a = gcd(ra1, dec1, ra2, dec2)
        pa = bear(ra1, dec1, ra2, dec2)
        return ra1, dec1, a, pa

    def sky2pix_ellipse(self, pos, a, b, pa):
        """
        Convert an ellipse from sky to pixel coordinates.

        Parameters
        ----------
        pos : (float, float)
            The (ra, dec) of the ellipse center (degrees).
        a, b, pa: float
            The semi-major axis, semi-minor axis and position angle of the ellipse (degrees).

        Returns
        -------
        x, y : (float, float)
            The (x, y) pixel coordinates of the ellipse center.
        sx, sy : (float, float)
            The major and minor axes (FWHM) in pixels.
        theta : float
            The rotation angle of the ellipse (degrees).
            theta = 0 corresponds to the ellipse being aligned with the x-axis.

        """
        ra, dec = pos
        x, y = self.sky2pix(pos)

        x_off, y_off = self.sky2pix(translate(ra, dec, a, pa))
        sx = np.hypot((x - x_off), (y - y_off))
        theta = np.arctan2((y_off - y), (x_off - x))

        x_off, y_off = self.sky2pix(translate(ra, dec, b, pa - 90))
        sy = np.hypot((x - x_off), (y - y_off))
        theta2 = np.arctan2((y_off - y), (x_off - x)) - np.pi / 2

        # The a/b vectors are perpendicular in sky space, but not always in pixel space
        # so we have to account for this by calculating the angle between the two vectors
        # and modifying the minor axis length
        defect = theta - theta2
        sy *= abs(np.cos(defect))

        return x, y, sx, sy, np.degrees(theta)

    def pix2sky_ellipse(self, pixel, sx, sy, theta):
        """
        Convert an ellipse from pixel to sky coordinates.

        Parameters
        ----------
        pixel : (float, float)
            The (x, y) coordinates of the center of the ellipse.
        sx, sy : float
            The major and minor axes (FHWM) of the ellipse, in pixels.
        theta : float
            The rotation angle of the ellipse (degrees).
            theta = 0 corresponds to the ellipse being aligned with the x-axis.

        Returns
        -------
        ra, dec : (float, float)
            The (ra, dec) coordinates of the center of the ellipse (degrees).

        a, b : (float, float)
            The semi-major and semi-minor axis of the ellipse (degrees).

        pa : float
            The position angle of the ellipse (degrees).
        """
        ra, dec = self.pix2sky(pixel)
        x, y = pixel
        v_sx = (x + sx * np.cos(np.radians(theta)),
                y + sx * np.sin(np.radians(theta)))
        ra2, dec2 = self.pix2sky(v_sx)
        major = gcd(ra, dec, ra2, dec2)
        pa = bear(ra, dec, ra2, dec2)

        v_sy = (x + sy * np.cos(np.radians(theta - 90)),
                y + sy * np.sin(np.radians(theta - 90)))
        ra2, dec2 = self.pix2sky(v_sy)
        minor = gcd(ra, dec, ra2, dec2)
        pa2 = bear(ra, dec, ra2, dec2) - 90

        # The a/b vectors are perpendicular in sky space, but not always in pixel space
        # so we have to account for this by calculating the angle between the two vectors
        # and modifying the minor axis length
        defect = pa - pa2
        minor *= abs(np.cos(np.radians(defect)))
        return ra, dec, major, minor, pa

    def get_pixbeam_pixel(self, x, y):
        """
        Determine the beam in pixels at the given location in pixel coordinates.

        Parameters
        ----------
        x , y : float
            The pixel coordinates at which the beam is determined.

        Returns
        -------
        beam : :class:`AegeanTools.wcs_helpers.Beam`
            A beam object, with a/b/pa in pixel coordinates.
        """
        ra, dec = self.pix2sky((x, y))
        return self.get_pixbeam(ra, dec)

    def get_beam(self, ra, dec):
        """
        Determine the beam at the given sky location.

        Parameters
        ----------
        ra, dec : float
            The sky coordinates at which the beam is determined.

        Returns
        -------
        beam : :class:`AegeanTools.wcs_helpers.Beam`
            A beam object, with a/b/pa in sky coordinates
        """
        # check to see if we need to scale the major axis based on the declination
        if self.lat is None:
            factor = 1
        else:
            # this works if the pa is zero. For non-zero pa it's a little more difficult
            factor = np.cos(np.radians(dec - self.lat))
        return Beam(self.beam.a / factor, self.beam.b, self.beam.pa)

    def get_pixbeam(self, ra, dec):
        """
        Determine the beam in pixels at the given location in sky coordinates.

        Parameters
        ----------
        ra , dec : float
            The sly coordinates at which the beam is determined.

        Returns
        -------
        beam : :class:`AegeanTools.wcs_helpers.Beam`
            A beam object, with a/b/pa in pixel coordinates.
        """

        if ra is None:
            ra, dec = self.pix2sky(self.refpix)
        pos = [ra, dec]

        beam = self.get_beam(ra, dec)
        _, _, major, minor, theta = self.sky2pix_ellipse(pos, beam.a, beam.b, beam.pa)

        if major < minor:
            major, minor = minor, major
            theta -= 90
            if theta < -180:
                theta += 180
        if not np.isfinite(theta):
            theta = 0
        if not all(np.isfinite([major, minor, theta])):
            beam = None
        else:
            beam = Beam(major, minor, theta)
        return beam

    def get_beamarea_deg2(self, ra, dec):
        """
        Calculate the area of the synthesized beam in square degrees.

        Parameters
        ----------
        ra, dec : float
            The sky coordinates at which the calculation is made.

        Returns
        -------
        area : float
            The beam area in square degrees.
        """
        barea = abs(self.beam.a * self.beam.b * np.pi)  # in deg**2 at reference coords
        if self.lat is not None:
            barea /= np.cos(np.radians(dec - self.lat))
        return barea

    def get_beamarea_pix(self, ra, dec):
        """
        Calculate the beam area in square pixels.

        Parameters
        ----------
        ra, dec : float
            The sky coordinates at which the calculation is made
        dec

        Returns
        -------
        area : float
            The beam area in square pixels.
        """
        parea = abs(self.pixscale[0] * self.pixscale[1])  # in deg**2 at reference coords
        barea = self.get_beamarea_deg2(ra, dec)
        return barea / parea

    def sky_sep(self, pix1, pix2):
        """
        calculate the GCD sky separation (degrees) between two pixels.

        Parameters
        ----------
        pix1, pix2 : (float, float)
            The (x,y) pixel coordinates for the two positions.

        Returns
        -------
        dist : float
            The distance between the two points (degrees).
        """
        pos1 = self.pix2sky(pix1)
        pos2 = self.pix2sky(pix2)
        sep = gcd(pos1[0], pos1[1], pos2[0], pos2[1])
        return sep


class PSFHelper(WCSHelper):
    """
    An extension of the :class:`AegeanTools.wcs_helpers.WCSHelper` class that also includes information about the
    point spread function over the sky.

    This allows for the distinction between the point spread function (what a point source looks like) and
    the synthesized beam in an image. These two can differ at low radio frequencies when the ionosphere is
    not playing nice.
    """

    # This __init__ overwrites that of the parent class without calling 'super'.
    # It might be naughty but it beats rewriting many of the get_X functions that I want to replicate.
    def __init__(self, psffile, wcshelper):
        if psffile is None:  # in this case this class should be transparent
            data = None
            wcs = wcshelper.wcs
        else:
            log.info("Loading PSF data from {0}".format(psffile))
            header = fits.getheader(psffile)
            data = fits.getdata(psffile)
            # the psf image has to have three dimensions
            # and they need to be ra/dec/beam
            if len(data.shape) != 3:
                log.critical("PSF file needs to have 3 dimensions, only {0} found".format(len(data.shape)))
                raise Exception("Invalid PSF file {0}".format(psffile))
            try:
                wcs = pywcs.WCS(header, naxis=2)
            except:
                wcs = pywcs.WCS(str(header), naxis=2)
        self.wcshelper = wcshelper
        self.data = data
        self.wcs = wcs

    def get_psf_sky(self, ra, dec):
        """
        Determine the local psf at a given sky location.
        The psf is returned in degrees.


        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        a, b, pa : float
            The psf semi-major axis, semi-minor axis, and position angle in (degrees).
            If a psf is defined then it is the psf that is returned, otherwise the image
            restoring beam is returned.
        """
        # If we don't have a psf map then we just fall back to using the beam
        # from the fits header (including ZA scaling)
        if self.data is None:
            beam = self.wcshelper.get_beam(ra, dec)
            return beam.a, beam.b, beam.pa

        x, y = self.sky2pix([ra, dec])
        # We leave the interpolation in the hands of whoever is making these images
        # clamping the x,y coords at the image boundaries just makes sense
        x = int(np.clip(x, 0, self.data.shape[1] - 1))
        y = int(np.clip(y, 0, self.data.shape[2] - 1))
        psf_sky = self.data[:, x, y]
        return psf_sky

    def get_psf_pix(self, ra, dec):
        """
        Determine the local psf (a,b,pa) at a given sky location.
        The psf is in pixel coordinates.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).


        Returns
        -------
        a, b, pa : float
            The psf semi-major axis (pixels), semi-minor axis (pixels), and rotation angle (degrees).
            If a psf is defined then it is the psf that is returned, otherwise the image
            restoring beam is returned.

        """
        psf_sky = self.get_psf_sky(ra, dec)
        psf_pix = self.wcshelper.sky2pix_ellipse([ra, dec], psf_sky[0], psf_sky[1], psf_sky[2])[2:]
        return psf_pix

    def get_pixbeam_pixel(self, x, y):
        """
        Get the psf at the location specified in pixel coordinates.
        The psf is also in pixel coordinates.

        Parameters
        ----------
        x, y : float
            The image position in pixels.

        Returns
        -------
        a, b, pa : float
            The psf semi-major axis (pixels), semi-minor axis (pixels), and rotation angle (degrees).
            If a psf is defined then it is the psf that is returned, otherwise the image
            restoring beam is returned.

        """
        # overriding the WCSHelper function of the same name means that we now calculate the
        # psf at the coordinates of the x/y pixel in the image WCS, rather than the psfimage WCS
        ra, dec = self.wcshelper.pix2sky([x, y])
        return self.get_pixbeam(ra, dec)

    def get_pixbeam(self, ra, dec):
        """
        Get the psf at the location specified in pixel coordinates.
        The psf is also in pixel coordinates.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        a, b, pa : float
            The psf semi-major axis (pixels), semi-minor axis (pixels), and rotation angle (degrees).
            If a psf is defined then it is the psf that is returned, otherwise the image
            restoring beam is returned.

        """
        # If there is no psf image then just use the fits header (plus lat scaling) from the wcshelper
        if self.data is None:
            return self.wcshelper.get_pixbeam(ra, dec)
        # get the beam from the psf image data
        psf = self.get_psf_pix(ra, dec)
        if not np.all(np.isfinite(psf)):
            log.warning("PSF requested, returned Null")
            return None
        return Beam(psf[0], psf[1], psf[2])

    def get_beam(self, ra, dec):
        """
        Get the psf as a :class:`AegeanTools.fits_image.Beam` object.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        beam : :class:`AegeanTools.wcs_helpers.Beam`
            The psf at the given location.
        """
        if self.data is None:
            return self.wcshelper.beam
        else:
            psf = self.get_psf_sky(ra, dec)
            if not all(np.isfinite(psf)):
                return None
            return Beam(psf[0], psf[1], psf[2])

    def get_beamarea_pix(self, ra, dec):
        """
        Calculate the area of the beam in square pixels.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        area : float
            The area of the beam in square pixels.
        """

        parea = abs(self.wcshelper.pixscale[0] * self.wcshelper.pixscale[1])  # in deg**2 at reference coords
        barea = self.get_beamarea_deg2(ra, dec)
        return barea / parea

    def get_beamarea_deg2(self, ra, dec):

        """
        Calculate the area of the beam in square degrees.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        area : float
            The area of the beam in square degrees.
        """
        beam = self.get_beam(ra, dec)
        if beam is None:
            return 0
        return beam.a * beam.b * np.pi

class Beam(object):
    """
    Small class to hold the properties of the beam.
    Properties are a,b,pa. No assumptions are made as to the units, but both a and b have to be >0.
    """
    def __init__(self, a, b, pa):
        if not (a > 0): raise AssertionError("major axis must be >0")
        if not (b > 0): raise AssertionError("minor axis must be >0")
        self.a = a
        self.b = b
        self.pa = pa

    def __str__(self):
        return "a={0} b={1} pa={2}".format(self.a, self.b, self.pa)



def get_pixinfo(header):
    """
    Return some pixel information based on the given hdu header
    pixarea - the area of a single pixel in deg2
    pixscale - the side lengths of a pixel (assuming they are square)

    Parameters
    ----------
    header : :class:`astropy.io.fits.HDUHeader`
        FITS header information

    Returns
    -------
    pixarea : float
        The are of a single pixel at the reference location, in square degrees.

    pixscale : (float, float)
        The pixel scale in degrees, at the reference location.

    Notes
    -----
    The reference location is not always at the image center, and the pixel scale/area may
    change over the image, depending on the projection.
    """
    if all(a in header for a in ["CDELT1", "CDELT2"]):
        pixarea = abs(header["CDELT1"]*header["CDELT2"])
        pixscale = (header["CDELT1"], header["CDELT2"])
    elif all(a in header for a in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]):
        pixarea = abs(header["CD1_1"]*header["CD2_2"]
                    - header["CD1_2"]*header["CD2_1"])
        pixscale = (header["CD1_1"], header["CD2_2"])
        if not (header["CD1_2"] == 0 and header["CD2_1"] == 0):
            log.warning("Pixels don't appear to be square -> pixscale is wrong")
    elif all(a in header for a in ["CD1_1", "CD2_2"]):
        pixarea = abs(header["CD1_1"]*header["CD2_2"])
        pixscale = (header["CD1_1"], header["CD2_2"])
    else:
        log.critical("cannot determine pixel area, using zero EVEN THOUGH THIS IS WRONG!")
        pixarea = 0
        pixscale = (0, 0)
    return pixarea, pixscale


def get_beam(header):
    """
    Create a :class:`AegeanTools.wcs_helpers.Beam` object from a fits header.

    BPA may be missing but will be assumed to be zero.

    if BMAJ or BMIN are missing then return None instead of a beam object.

    Parameters
    ----------
    header : :class:`astropy.io.fits.HDUHeader`
        The fits header.

    Returns
    -------
    beam : :class:`AegeanTools.wcs_helpers.Beam`
        Beam object, with a, b, and pa in degrees.
    """

    if "BPA" not in header:
        log.warning("BPA not present in fits header, using 0")
        bpa = 0
    else:
        bpa = header["BPA"]

    if "BMAJ" not in header:
        log.warning("BMAJ not present in fits header.")
        bmaj = None
    else:
        bmaj = header["BMAJ"]

    if "BMIN" not in header:
        log.warning("BMIN not present in fits header.")
        bmin = None
    else:
        bmin = header["BMIN"]
    if None in [bmaj, bmin, bpa]:
        return None
    beam = Beam(bmaj, bmin, bpa)
    return beam


def fix_aips_header(header):
    """
    Search through an image header. If the keywords BMAJ/BMIN/BPA are not set,
    but there are AIPS history cards, then we can populate the BMAJ/BMIN/BPA.
    Fix the header if possible, otherwise don't. Either way, don't complain.


    Parameters
    ----------
    header : :class:`astropy.io.fits.HDUHeader`
        Fits header which may or may not have AIPS history cards.

    Returns
    -------
    header : :class:`astropy.io.fits.HDUHeader`
        A header which has BMAJ, BMIN, and BPA keys, as well as a new HISTORY card.
    """
    if 'BMAJ' in header and 'BMIN' in header and 'BPA' in header:
        # The header already has the required keys so there is nothing to do
        return header
    aips_hist = [a for a in header['HISTORY'] if a.startswith("AIPS")]
    if len(aips_hist) == 0:
        # There are no AIPS history items to process
        return header
    for a in aips_hist:
        if "BMAJ" in a:
            # this line looks like
            # 'AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00'
            words = a.split()
            bmaj = float(words[3])
            bmin = float(words[5])
            bpa = float(words[7])
            break
    else:
        # there are AIPS cards but there is no BMAJ/BMIN/BPA
        return header
    header['BMAJ'] = bmaj
    header['BMIN'] = bmin
    header['BPA'] = bpa
    header['HISTORY'] = 'Beam information AIPS->fits by AegeanTools'
    return header




