#! /usr/bin/env python
"""

"""
__author__ = 'Paul Hancock'

import numpy as np
import math

from angle_tools import gcd, bear, translate
from AegeanTools.fits_image import Beam, get_beam, get_pixinfo

# the glory of astropy
import astropy
import astropy.wcs as pywcs
from astropy.io import fits

# join the Aegean logger
import logging
log = logging.getLogger('Aegean')



class WCSHelper(object):

    @classmethod
    def from_header(cls, header, beam=None):
        """
        Create a new WCSHelper class from the given header
        This will not set the latitude of the telesocpe so this needs to be set by the user
        if it is needed
        :param header: HDUHeader
        :return: a WCSHelper object
        """
        try:
            wcs = pywcs.WCS(header, naxis=2)
        except:
            wcs = pywcs.WCS(str(header), naxis=2)

        if beam is None:
            beam = get_beam(header)
        else:
            beam = beam

        if beam is None:
            logging.critical("Cannot determine beam information")

        _, pixscale = get_pixinfo(header)
        refpix = (header['CRPIX1'],header['CRPIX2'])
        return cls(wcs, beam, pixscale, refpix)

    @classmethod
    def from_file(cls, filename, beam=None):
        """
        Create a new WCSHelper class from a given fits file
        :param filename:
        :param beam:
        :return:
        """
        header = fits.getheader(filename)
        return cls.from_header(header,beam)

    def __init__(self, wcs, beam, pixscale, refpix, lat=None):
        self.wcs = wcs
        self.beam = beam
        self.pixscale = pixscale
        self.refpix = refpix
        self.lat = lat


    def pix2sky(self, pixel):
        """
        Take pixel=(x,y) coords
        convert to pos=(ra,dec) coords
        """
        x, y = pixel
        #wcs and pyfits have oposite ideas of x/y
        return self.wcs.wcs_pix2world([[y, x]], 1)[0]


    def sky2pix(self, pos):
        """
        Take pos = (ra,dec) coords
        convert to pixel = (x,y) coords
        """
        pixel = self.wcs.wcs_world2pix([pos], 1)
        #wcs and pyfits have oposite ideas of x/y
        return [pixel[0][1], pixel[0][0]]


    def sky2pix_vec(self, pos, r, pa):
        """Convert a vector from sky to pixel corrds
        vector is calculated at an origin pos=(ra,dec)
        and has a magnitude (r) [in degrees]
        and an angle (pa) [in degrees]
        input:
            pos - (ra,dec) of vector origin
            r - magnitude in degrees
            pa - angle in degrees
        return:
        x,y - corresponding to position ra,dec
        r,theta - magnitude (pixels) and angle (degrees) of the original vector
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
        Convert a vector from pixel to sky coords
        vector is calculated at an origin pixel=(x,y)
        and has a magnitude (r) [in pixels]
        and an angle (theta) [in degrees]
        input:
            pixel - (x,y) of origin
            r - magnitude in pixels
            theta - in degrees
        return:
        ra,dec - corresponding to pixels x,y
        r,pa - magnitude and angle (degrees) of the original vector, as measured on the sky
        """
        ra1, dec1 = self.pix2sky(pixel)
        x, y = pixel
        a = [x + r * np.cos(np.radians(theta)),
             y + r * np.sin(np.radians(theta))]
        locations = self.pix2sky(a)
        ra2, dec2 = locations
        a = gcd(ra1, dec1, ra2, dec2)
        pa = bear(ra1, dec1, ra2, dec2)
        return ra1, dec1, a, pa


    def get_pixbeam_pixel(self, x, y):
        """
        A wrapper around get_pixbeam for when you only know the pixel coords
        :param x:
        :param y:
        :return:
        """
        ra,dec = self.pix2sky([x,y])
        return self.get_pixbeam(ra,dec)


    def get_pixbeam(self, ra, dec):
        """
        Use global_data to get beam (sky scale), and img.pixscale.
        Calculate a beam in pixel scale, pa is always zero
        :return: A beam in pixel scale
        """
        # TODO: update this to incorporate elevation scaling when needed

        if ra is None:
            ra,dec = self.pix2sky(self.refpix)
        pos = [ra,dec]

        major = abs(self.beam.a/(self.pixscale[0]*math.sin(math.radians(self.beam.pa)) +
                                 self.pixscale[1]*math.cos(math.radians(self.beam.pa)) ))

        minor = abs(self.beam.b/(self.pixscale[1]*math.sin(math.radians(self.beam.pa)) +
                                 self.pixscale[0]*math.cos(math.radians(self.beam.pa)) ))

        theta =  self.sky2pix_vec(pos, self.beam.a, self.beam.pa)[3]

        if major<minor:
            major,minor = minor,major
            theta -=90
            if theta < -180:
                theta += 180
        if not np.isfinite(theta):
            theta = 0
        beam = Beam(major, minor, theta)
        return beam


    def get_beamarea_deg2(self, ra, dec):
        """

        :param ra:
        :param dec:
        :return:
        """
        barea = abs(self.beam.a * self.beam.b * np.pi) # in deg**2 at reference coords
        if self.lat is not None:
            barea /= np.cos(np.radians(dec-self.lat))
        return barea


    def get_beamarea_pix(self, ra, dec):
        """
        Calculate the area of the beam at a given location
        scale area based on elevation if the telescope latitude is known.
        :param ra:
        :param dec:
        :return:
        """
        parea = abs(self.pixscale[0] * self.pixscale[1]) # in deg**2 at reference coords
        barea = self.get_beamarea_deg2(ra,dec)
        return barea/parea


    def sky_sep(self, pix1, pix2):
        """
        calculate the sky separation between two pixels
        Input:
            pix1 = [x1,y1]
            pix2 = [x2,y2]
        Returns:
            sep = separation in degrees
        """
        pos1 = self.pix2sky(pix1)
        pos2 = self.pix2sky(pix2)
        sep = gcd(pos1[0], pos1[1], pos2[0], pos2[1])
        return sep


class WCSHelperTest(object):
    """
    A test class for WCSHelper
    """
    def __init__(self):
        self.helper = WCSHelper.from_file('Test/Images/1904-66_SIN.fits')
        self.vector_test()

    def vector_test(self):
        print "Testing vector round trip... ",
        initial = [1,45] #r,theta = 1,45 (degrees)
        ref = self.helper.refpix
        ra,dec,dist,ang = self.helper.pix2sky_vec(ref, *initial)
        x,y,r,theta = self.helper.sky2pix_vec([ra,dec], dist, ang)
        print "Start: x {0}, y {1}, r {2}, theta {3}".format(ref[0],ref[1],*initial)
        print "sky: ra {0}, dec {1}, dist {2}, ang {3}".format(ra,dec,dist,ang)
        print "Final: x {0}, y {1}, r {2}, theta {3}".format(x,y,r,theta)
        if abs(r-initial[0])<1e-9 and abs(theta-initial[1])<1e-9:
            print "Pass"
            return True
        else:
            print "Fail"
            return False


class PSFHelper(object):
    """
    A class that will store information about the PSF, which is assumed to be direction dependent.
    """
    def __init__(self, data, wcshelper):
        self.wcshelper = wcshelper
        self.data = data


    def get_pixbeam(self,ra,dec):
        """
        Get the beam shape at this location.
        :param ra:
        :param dec:
        :return:
        """
        if self.data is None:
            return self.wcshelper.get_pixbeam()

class PSFHelperTest(object):
    """

    """
    def __init__(self):
        psffile = "Test/Images/1904_66_SIN_psf.fits"
        wcsfile = "Test/Images/1904_66_SIN.fits"
        psfdata = fits.getdata(psffile)
        self.helper = PSFHelper(psfdata,WCSHelper.from_file(wcsfile))
