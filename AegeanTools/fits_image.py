#! /usr/bin/env python
from __future__ import print_function

__author__= "Paul Hancock"

import numpy
import astropy.wcs as pywcs
import scipy.stats
import logging
from .fits_interp import expand

# Join the Aegean logger
log = logging.getLogger("Aegean")


def get_pixinfo(header):
    """
    Return some pixel information based on the given hdu header
    pixarea - the area of a single pixel in deg2
    pixscale - the side lengths of a pixel (assuming they are square)

    Parameters
    ----------
    header : HDUHeader or dict
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
    Create a :class:`AegeanTools.fits_image.Beam` object from a fits header.

    BPA may be missing but will be assumed to be zero.

    if BMAJ or BMIN are missing then return None instead of a beam object.

    Parameters
    ----------
    header : HDUHeader
        The fits header.

    Returns
    -------
    beam : :class:`AegeanTools.fits_image.Beam`
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
    header : HDUHeader
        Fits header which may or may not have AIPS history cards.

    Returns
    -------
    header : HDUHeader
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


class FitsImage(object):
    """
    An object that handles the loading and manipulation of a fits file.
    """

    def __init__(self, filename=None, hdu_index=0, beam=None, slice=None):
        """
        Parameters
        ----------
        filename : str or astropy.io.fits.HDUList
            The name of the fits image or an already loaded HDUList

        hdu_index : int
            The index of the FITS HDU. Default = 0.

        beam : Beam
            The synthesized beam for this image, using sky coordinates.
            If beam is None then it will be created from the fits header.
            Default = None.

        slice : int
            If the input data has 3 dimensions then this will specify the index into the 3rd dimension
            which will be extracted as the image.
            Default = None.
        """

        self.hdu = expand(filename)[hdu_index] # auto detects if the file needs expanding

        self._header = self.hdu.header
        # need to read these headers before we 'touch' the data or they dissappear
        if "BZERO" in self._header:
            self.bzero = self._header["BZERO"]
        else:
            self.bzero = 0
        if "BSCALE" in self._header:
            self.bscale = self._header["BSCALE"]
        else:
            self.bscale = 1

        self.filename = filename
        # fix possible problems with miriad generated fits files % HT John Morgan.
        try:
            self.wcs = pywcs.WCS(self._header, naxis=2)
        except:
            self.wcs = pywcs.WCS(str(self._header), naxis=2)
            
        self.x = self._header['NAXIS1']
        self.y = self._header['NAXIS2']

        self.pixarea, self.pixscale = get_pixinfo(self._header)

        if beam is None:
            self.beam = get_beam(self._header)
            if self.beam is None:
                log.critical("Beam info is not in fits header.")
                log.critical("Beam info not supplied by user. Stopping.")
                raise Exception("Unable to determine beam.")
        else:  # use the supplied beam
            self.beam = beam
        self._rms = None
        self._pixels = numpy.squeeze(self.hdu.data)
        # if we have a fits cube just use a single slice
        if len(self._pixels.shape) == 3:
            if slice is None:
                log.critical("Image is a cube, but no slice is given")
                raise Exception("Image is a cube, but no slice is given")
            log.info("Image is a cube, using slice {0}".format(slice))
            self._pixels = self._pixels[slice, :, :]
        elif len(self._pixels.shape) > 3:
            log.critical("Image has >3 axes.")
            raise Exception("Images with >3 axes not supported.")
        # convert +/- inf to nan
        self._pixels[numpy.where(numpy.isinf(self._pixels))] = numpy.nan
        # del self.hdu
        log.debug("Using axes {0} and {1}".format(self._header['CTYPE1'], self._header['CTYPE2']))

    def get_pixels(self):
        """
        Get the image data.

        Returns
        -------
        pixels : numpy.ndarray
            2d Array of image pixels.
        """
        return self._pixels

    def set_pixels(self, pixels):
        """
        Set the image data.
        Will not work if the new image has a different shape than the current image.

        Parameters
        ----------
        pixels : numpy.ndarray
            New image data

        Returns
        -------
        None
        """
        if not (pixels.shape == self._pixels.shape):
            raise AssertionError("Shape mismatch between pixels supplied {0} and existing image pixels {1}".format(pixels.shape,self._pixels.shape))
        self._pixels = pixels
        # reset this so that it is calculated next time the function is called
        self._rms = None
        return

    def get_background_rms(self):
        """
        Calculate the rms of the image. The rms is calculated from the interqurtile range (IQR), to
        reduce bias from source pixels.

        Returns
        -------
        rms : float
            The image rms.

        Notes
        -----
        The rms value is cached after first calculation.
        """
        # TODO: return a proper background RMS ignoring the sources
        # This is an approximate method suggested by PaulH.
        # I have no idea where this magic 1.34896 number comes from...
        if self._rms is None:
            # Get the pixels values without the NaNs
            data = numpy.extract(self.hdu.data > -9999999, self.hdu.data)
            p25 = scipy.stats.scoreatpercentile(data, 25)
            p75 = scipy.stats.scoreatpercentile(data, 75)
            iqr = p75 - p25
            self._rms = iqr / 1.34896
        return self._rms
    
    def pix2sky(self, pixel):
        """
        Get the sky coordinates for a given image pixel.

        Parameters
        ----------
        pixel : (float, float)
            Image coordinates.

        Returns
        -------
        ra,dec : float
            Sky coordinates (degrees)

        """
        pixbox = numpy.array([pixel, pixel])
        skybox = self.wcs.all_pix2world(pixbox, 1)
        return [float(skybox[0][0]), float(skybox[0][1])]

    def get_hdu_header(self):
        """
        Get the image header.
        """
        return self._header

    def sky2pix(self, skypos):
        """
        Get the pixel coordinates for a given sky position (degrees).

        Parameters
        ----------
        skypos : (float,float)
            ra,dec position in degrees.

        Returns
        -------
        x,y : float
            Pixel coordinates.

        """
        """
        Get the pixel coordinates [x,y] (floats) given skypos [ra,dec] (degrees)
        """
        skybox = [skypos, skypos]
        pixbox = self.wcs.all_world2pix(skybox, 1)
        return [float(pixbox[0][0]), float(pixbox[0][1])]


class Beam(object):
    """
    Small class to hold the properties of the beam.
    Properties are a,b,pa. No assumptions are made as to the units, but both a and b have to be >0.
    """
    def __init__(self, a, b, pa, pixa=None, pixb=None):
        if not (a > 0): raise AssertionError("major axis must be >0")
        if not (b > 0): raise AssertionError("minor axis must be >0")
        self.a = a
        self.b = b
        self.pa = pa
        self.pixa = pixa
        self.pixb = pixb
    
    def __str__(self):
        return "a={0} b={1} pa={2}".format(self.a, self.b, self.pa)
