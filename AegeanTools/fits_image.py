"""
Created on 18/07/2011

@author: jay
Modified by: Paul Hancock 2012 onwards
"""

import numpy
import astropy.wcs as pywcs
import scipy.stats
import logging
import sys
from fits_interp import expand

# Join the Aegean logger
log = logging.getLogger("Aegean")


def get_pixinfo(header):
    """
    Return some pixel information based on the given hdu header
    pixarea - the area of a single pixel in deg2
    pixscale - the side lengths of a pixel (assuming they are square)
    :param header: HDUHeader
    :return: pixarea, pixscale
    """
    # this is correct at the center of the image for all images, and everywhere for conformal projections
    if all(a in header for a in ["CDELT1", "CDELT2"]):
        pixarea = abs(header["CDELT1"]*header["CDELT2"])
        pixscale = (header["CDELT1"], header["CDELT2"])
    elif all(a in header for a in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]):
        pixarea = abs(header["CD1_1"]*header["CD2_2"]
                    - header["CD1_2"]*header["CD2_1"])
        pixscale = (header["CD1_1"], header["CD2_2"])
        if not (header["CD1_2"] == 0 and header["CD2_1"] == 0):
            log.warn("Pixels don't appear to be square -> pixscale is wrong")
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
    Read the supplied fits header and extract the beam information
    BPA may be missing but will be assumed to be zero
    if BMAJ or BMIN are missing then return None instead of a beam object
    :param header: HDUheader
    :return: a Beam object or None
    """

    if "BPA" not in header:
        log.warn("BPA not present in fits header, using 0")
        bpa = 0
    else:
        bpa = header["BPA"]

    if "BMAJ" not in header:
        log.warn("BMAJ not present in fits header.")
        bmaj = None
    else:
        bmaj = header["BMAJ"]

    if "BMIN" not in header:
        log.warn("BMIN not present in fits header.")
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
    :param header:
    :return:
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

class FitsImage():
    """
    An object that handles the loading and manipulation of a fits file,
    """

    def __init__(self, filename=None, hdu_index=0, beam=None):
        """
        filename: the name of the fits image file or an instance of astropy.io.fits.HDUList
        hdu_index = index of FITS HDU when extensions are used (0 is primary HDU)
        hdu = a pyfits hdu. if provided the object is constructed from this instead of
              opening the file (filename is ignored)  
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
                sys.exit(1)
        else:  # use the supplied beam
            self.beam = beam
        self._rms = None
        self._pixels = numpy.squeeze(self.hdu.data)
        # convert +/- inf to nan
        self._pixels[numpy.where(numpy.isinf(self._pixels))] = numpy.nan
        # del self.hdu
        log.debug("Using axes {0} and {1}".format(self._header['CTYPE1'], self._header['CTYPE2']))

    def get_pixels(self):
        return self._pixels

    def set_pixels(self, pixels):
        """
        Allow the pixels to be replaced
        Will only work if pixels.shape is the same as self._pixels.shape
        """
        assert pixels.shape == self._pixels.shape, "Shape mismatch between pixels supplied {0} and existing image pixels {1}".format(pixels.shape,self._pixels.shape)
        self._pixels = pixels
            
    def get_background_rms(self):
        """
        Return the background RMS (Jy)
        NB - value is calculated on first request then cached for speed
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
        Get the sky coordinates [ra,dec] (degrees) given pixel [x,y] (float)
        """
        pixbox = numpy.array([pixel, pixel])
        skybox = self.wcs.wcs_pix2sky(pixbox, 1)
        return [float(skybox[0][0]), float(skybox[0][1])]

    def get_hdu_header(self):
        return self._header

    def sky2pix(self, skypos):
        """
        Get the pixel coordinates [x,y] (floats) given skypos [ra,dec] (degrees)
        """
        skybox = [skypos, skypos]
        pixbox = self.wcs.wcs_sky2pix(skybox, 1)
        return [float(pixbox[0][0]), float(pixbox[0][1])] 


class Beam():
    """
    Small class to hold the properties of the beam
    """
    def __init__(self, a, b, pa, pixa=None, pixb=None):
        assert a > 0, "major axis must be >0"
        assert b > 0, "minor axis must be >0"
        self.a = a
        self.b = b
        self.pa = pa
        self.pixa = pixa
        self.pixb = pixb
    
    def __str__(self):
        return "a={0} b={1} pa={2}".format(self.a, self.b, self.pa)
