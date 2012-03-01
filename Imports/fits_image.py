'''
Created on 18/07/2011

@author: jay
'''

import pyfits
import numpy
import pywcs
import scipy.stats
import logging
from math import pi

class FitsImage():
    def __init__(self, filename, hdu_index=0, hdu=None):
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
        self.bzero= self.hdu.header["BZERO"]
        self.bscale=self.hdu.header["BSCALE"]
        
        self.filename = filename
        self.wcs = pywcs.WCS(self.hdu.header, naxis=2)
        self.x = self.hdu.header['NAXIS1']
        self.y = self.hdu.header['NAXIS2']
        self.deg_per_pixel_x = self.hdu.header["CDELT1"] # is this always right?
        self.deg_per_pixel_y = self.hdu.header["CDELT2"] # is this always right?
        if not self.hdu.header.has_key("BPA"): #if the bpa isn't specified add it as zero
            bpa=0
        else:
            bpa=self.hdu.header["BPA"]*pi/180
        # TODO: handle non-square pixels and elliptical beam
        ## for now: elliptical beams are replacedwith a circular beam the size of the minor axis.
        self.beam=Beam(abs(self.hdu.header["BMAJ"]/self.deg_per_pixel_y),
                       abs(self.hdu.header["BMIN"]/self.deg_per_pixel_x),
                       bpa)
        self.pixels_per_beam=min(self.beam.a,self.beam.b)
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
                self._pixels = numpy.nan_to_num(self.hdu.data)
            elif len(self.hdu.data.shape) == 4:
                self._pixels = numpy.nan_to_num(self.hdu.data[0][0])
            else:
                raise Exception("Can't handle {0} dimensions".format(len(self.hdu.data.shape)))
            
        #do we need to check for blank pixels?
        if float(pyfits.__version__[0:3])<2.3:
            #are there likely to be any blank pixels?
            if self.hdu.header.has_key("BLANK"):
                self.wrangle_nans()
        
        return self._pixels

    def wrangle_nans(self):
        '''
        For versions of pyfits <2.3 blank pixels are imported with crazy fluxes
        These need to be changed into NaN as per pyfits v2.3+
        '''
        blank = self.hdu.header["BLANK"]*self.bscale+self.bzero
        self._pixels[numpy.where(abs(self._pixels-blank)<1e-9)]=numpy.NaN
    
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

    def sky2pix(self, skypos):
        '''Get the pixel coordinates [x,y] (floats) given skypos [ra,dec] (degrees)'''
        skybox = [skypos, skypos]
        pixbox = self.wcs.wcs_sky2pix(skybox, 1)
        return [float(pixbox[0][0]), float(pixbox[0][1])] 


class Beam():
    """Small class to hold the properties of the primary beam"""
    def __init__(self,a,b,pa):
        self.a=abs(a)
        self.b=abs(b)
        self.pa=pa
    
    def __str__(self):
        return "a={0} b={1} pa={2}".format(self.a, self.b, self.pa)
