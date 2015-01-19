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
    def __init__(self, filename=None, hdu_index=0, hdu=None, beam=None):
        """
        filename: the name of the fits image file or an instance of astropy.io.fits.HDUList
        hdu_index = index of FITS HDU when extensions are used (0 is primary HDU)
        hdu = a pyfits hdu. if provided the object is constructed from this instead of
              opening the file (filename is ignored)  
        """
        if isinstance(filename,pyfits.HDUList):
            logging.debug("accepting already loaded file {0}".format(filename.filename()))
            self.hdu = filename[hdu_index]
        elif hdu:
            self.hdu = hdu
        else:
            logging.debug("Loading HDU {0} from {1}".format(hdu_index, filename))
            try:
                hdus = pyfits.open(filename)
            except IOError,e:
                if "END" in e.message:
                    logging.warn(e.message)
                logging.warn("trying to ignore this, but you should really fix it")
                hdus = pyfits.open(filename,ignore_missing_end=True)
            self.hdu = hdus[hdu_index]
            del hdus
        
        self._header=self.hdu.header
        #need to read these headers before we 'touch' the data or they dissappear
        if "BZERO" in self._header:
            self.bzero= self._header["BZERO"]
        else:
            self.bzero=0
        if "BSCALE" in self._header:
            self.bscale=self._header["BSCALE"]
        else:
            self.bscale=1
            
        self.filename = filename
        #fix possible problems with miriad generated fits files % HT John Morgan.
        try:
            self.wcs = pywcs.WCS(self._header, naxis=2)
        except:
            self.wcs = pywcs.WCS(str(self._header),naxis=2)
            
        self.x = self._header['NAXIS1']
        self.y = self._header['NAXIS2']
        #this is correct at the center of the image for all images, and everywhere for conformal projections
        if all( [ a in self._header for a in ["CDELT1","CDELT2"]]):
            self.pixarea = abs(self._header["CDELT1"]*self._header["CDELT2"])
            self.pixscale = (self._header["CDELT1"], self._header["CDELT2"])
        elif all( [a in self._header for a in ["CD1_1","CD1_2","CD2_1","CD2_2"]]):
            self.pixarea = abs( self._header["CD1_1"]*self._header["CD2_2"] - self._header["CD1_2"]*self._header["CD2_1"])
            self.pixscale = (self._header["CD1_1"], self._header["CD2_2"])
            if not (self._header["CD1_2"] ==0 and self._header["CD2_1"]==0):
                logging.warn("Pixels don't appear to be square -> pixscale is wrong")
        elif all([a in self._header for a in ["CD1_1","CD2_2"]]):
            self.pixarea = abs(self._header["CD1_1"]*self._header["CD2_2"])
            self.pixscale = (self._header["CD1_1"], self._header["CD2_2"])
        else:
            logging.warn("cannot determine pixel area, using zero EVEN THOUGH THIS IS WRONG!")
            self.pixarea = 0
            self.pixscale = (0,0)
        
        if beam is None:
            #if the bpa isn't specified add it as zero
            if "BPA" not in self._header:
                logging.info("BPA not present in fits header, using 0")
                bpa=0
            else:
                bpa=self._header["BPA"]
                
            if "BMAJ" not in self._header:
                logging.error("BMAJ not present in fits header.")
                logging.error("BMAJ not supplied by user. Exiting.")
                sys.exit(0)
            else:
                bmaj = self._header["BMAJ"]
                
            if "BMIN" not in self._header:
                logging.error("BMIN not present in fits header.")
                logging.error("BMIN not supplied by user. Exiting.")
                sys.exit(0)
            else:
                bmin = self._header["BMIN"]
            self.beam=Beam(bmaj, bmin, bpa)
        else: #use the supplied beam
            self.beam=beam
        self._rms = None
        self._pixels = numpy.squeeze(self.hdu.data)
        #convert +/- inf to nan
        self._pixels[numpy.where(numpy.isinf(self._pixels))] = numpy.nan
        #del self.hdu
        logging.debug("Using axes {0} and {1}".format(self._header['CTYPE1'],self._header['CTYPE2']))

        
    def get_pixels(self):
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
        return self._header

    def sky2pix(self, skypos):
        '''Get the pixel coordinates [x,y] (floats) given skypos [ra,dec] (degrees)'''
        skybox = [skypos, skypos]
        pixbox = self.wcs.wcs_sky2pix(skybox, 1)
        return [float(pixbox[0][0]), float(pixbox[0][1])] 


class Beam():
    """
    Small class to hold the properties of the beam
    """
    def __init__(self, a, b, pa, pixa=None, pixb=None):
        assert a>0, "major axis must be >0"
        assert b>0, "minor axis must be >0"
        self.a=a
        self.b=b
        self.pa=pa
        self.pixa=pixa
        self.pixb=pixb
    
    def __str__(self):
        return "a={0} b={1} pa={2}".format(self.a, self.b, self.pa)
