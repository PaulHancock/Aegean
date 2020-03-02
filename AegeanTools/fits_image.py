#! /usr/bin/env python
"""
Tools for interacting with fits images (HUDLists)
"""

from __future__ import print_function

import numpy
import astropy.wcs as pywcs
import scipy.stats
import logging
from .fits_interp import expand
from .wcs_helpers import get_pixinfo, get_beam

__author__= "Paul Hancock"

# Join the Aegean logger
log = logging.getLogger("Aegean")


class FitsImage(object):
    """
    An object that handles the loading and manipulation of a fits file.
    """

    def __init__(self, filename=None, hdu_index=0, beam=None, cube_index=None):
        """
        Parameters
        ----------
        filename : str or astropy.io.fits.HDUList
            The name of the fits image or an already loaded HDUList

        hdu_index : int
            The index of the FITS HDU. Default = 0.

        beam : AegeanTools.wcs_helpers.Beam
            The synthesized beam for this image, using sky coordinates.
            If beam is None then it will be created from the fits header.
            Default = None.

        cube_index : int
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
        except:  # TODO: figure out what error is being thrown
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
            if cube_index is None:
                log.critical("Image is a cube, but no cube_index is given")
                raise Exception("Image is a cube, but no cube_index is given")
            log.info("Image is a cube, using cube_index {0}".format(cube_index))
            self._pixels = self._pixels[cube_index, :, :]
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
        skybox = [skypos, skypos]
        pixbox = self.wcs.all_world2pix(skybox, 1)
        return [float(pixbox[0][0]), float(pixbox[0][1])]


