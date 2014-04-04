'''
Created on 18/07/2011

@author: jay
'''

import astropy.io.fits as pyfits
import numpy
import astropy.wcs as pywcs
import scipy.stats
import logging,sys
from math import pi,cos,sin,sqrt

class FitsImage():
    version='$Revision$'
    def __init__(self, filename, hdu_index=0, hdu=None, beam=None):
        """
        filename: the name of the fits image file
        hdu_index = index of FITS HDU when extensions are used (0 is primary HDU)
        hdu = a pyfits hdu. if provided the object is constructed from this instead of
              opening the file (filename is ignored)  
        """
        if hdu:
            self.hdu = hdu
        else:
            logging.debug("Loading HDU {0} from {1}".format(hdu_index, filename))
            hdus = pyfits.open(filename)
            self.hdu = hdus[hdu_index]
            
        #need to read these headers before we 'touch' the data or they dissappear
        if "BZERO" in self.hdu.header:
            self.bzero= self.hdu.header["BZERO"]
        else:
            self.bzero=0
        if "BSCALE" in self.hdu.header:
            self.bscale=self.hdu.header["BSCALE"]
        else:
            self.bscale=1
            
        self.filename = filename
        #fix possible problems with miriad generated fits files % HT John Morgan.
        try:
            self.wcs = pywcs.WCS(self.hdu.header, naxis=2)
        except:
            self.wcs = pywcs.WCS(str(self.hdu.header),naxis=2)
            
        self.x = self.hdu.header['NAXIS1']
        self.y = self.hdu.header['NAXIS2']
        #this is correct at the center of the image for all images, and everywhere for conformal projections
        self.pixarea = abs(self.hdu.header["CDELT1"]*self.hdu.header["CDELT2"])

        #the following no longer complies with the fits standard so i'm going to comment it out.
        #self.deg_per_pixel_x = self.hdu.header["CDELT1"] # is this always right?
        #self.deg_per_pixel_y = self.hdu.header["CDELT2"] # is this always right?
        
        if beam is None:
            #if the bpa isn't specified add it as zero
            if "BPA" not in self.hdu.header:
                logging.info("BPA not present in fits header, using 0")
                bpa=0
            else:
                bpa=self.hdu.header["BPA"]
                
            if "BMAJ" not in self.hdu.header:
                logging.error("BMAJ not present in fits header.")
                logging.error("BMAJ not supplied by user. Exiting.")
                sys.exit(0)
            else:
                bmaj = self.hdu.header["BMAJ"]
                
            if "BMIN" not in self.hdu.header:
                logging.error("BMIN not present in fits header.")
                logging.error("BMIN not supplied by user. Exiting.")
                sys.exit(0)
            else:
                bmin = self.hdu.header["BMIN"]
            self.beam=Beam(bmaj, bmin, bpa)
        else: #use the supplied beam
            self.beam=beam
        self._pixels = None
        self._rms = None
        
    def get_pixels(self):
        '''
        Returns all pixel values.
        Returns a numpy array with [y,x] as per pyfits.
        NB - value is calculated on first request then cached for speed
        '''
        # FIXME: this is specific to MWA files which have frequency and stokes
        # dimensions of length 1
        if self._pixels is None:
            if len(self.hdu.data.shape) == 2:
                self._pixels = self.hdu.data
            elif len(self.hdu.data.shape) == 3:
                
                self._pixels = self.hdu.data[0]
            elif len(self.hdu.data.shape) == 4:
                self._pixels = self.hdu.data[0][0]
            else:
                raise Exception("Can't handle {0} dimensions".format(len(self.hdu.data.shape)))
            logging.debug("Using axes {0} and {1}".format(self.hdu.header['CTYPE1'],self.hdu.header['CTYPE2']))
        return self._pixels

    def set_pixels(self,pixels):
        """
        Allow the pixels to be replaced
        Will only work if pixels.shape is the same as self._pixels.shape
        """
        assert pixels.shape == self._pixels.shape, "Shape mismatch between pixels supplied {0} and existing image pixels {1}".format(pixels.shape,self._pixels.shape)
        self._pixels = pixels
            


    def get_background_rms(self):
        '''
        Return the background RMS (Jy)
        NB - value is calculated on first request then cached for speed
        '''
        #TODO: return a proper background RMS ignoring the sources
        # This is an approximate method suggested by PaulH.
        # I have no idea where this magic 1.34896 number comes from...
        if self._rms is None:
            # Get the pixels values without the NaNs
            data = numpy.extract(self.hdu.data>-9999999, self.hdu.data)
            p25 = scipy.stats.scoreatpercentile(data, 25)
            p75 = scipy.stats.scoreatpercentile(data, 75)
            iqr = p75 - p25
            self._rms = iqr / 1.34896
        return self._rms
    
    def pix2sky(self, pixel):
        '''Get the sky coordinates [ra,dec] (degrees) given pixel [x,y] (float)'''
        pixbox = numpy.array([pixel, pixel])
        skybox = self.wcs.wcs_pix2sky(pixbox, 1)
        return [float(skybox[0][0]), float(skybox[0][1])]

    def get_hdu_header(self):
        return self.hdu.header

    def sky2pix(self, skypos):
        '''Get the pixel coordinates [x,y] (floats) given skypos [ra,dec] (degrees)'''
        skybox = [skypos, skypos]
        pixbox = self.wcs.wcs_sky2pix(skybox, 1)
        return [float(pixbox[0][0]), float(pixbox[0][1])] 


class Beam():
    """
    Small class to hold the properties of the beam
    a/b in degrees
    pa in radians
    """
    def __init__(self,a,b,pa):
        assert a>0, "major axis must be >0"
        assert b>0, "minor axis must be >0"
        self.a=a
        self.b=b
        self.pa=pa
    
    def __str__(self):
        return "a={0} b={1} pa={2}".format(self.a, self.b, self.pa)
