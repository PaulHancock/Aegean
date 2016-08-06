#! /usr/bin/env python
"""
This module contains two classes that provide WCS functions that are not
part of the WCS toolkit, as well as some wrappers around the provided tools
to make them a lot easier to use.
"""
__author__ = 'Paul Hancock'

import numpy as np

from angle_tools import gcd, bear, translate
from fits_image import Beam, get_beam, get_pixinfo

# the glory of astropy
import astropy
import astropy.wcs as pywcs
from astropy.io import fits

# join the Aegean logger
import logging
log = logging.getLogger('Aegean')


class WCSHelper(object):
    """
    A wrapper around astropy.wcs that provides extra functionality.
    Functionality hides the c/fortran indexing troubles, as well as providing:
    - sky2pix/pix2sky functions for vectors and ellipses.
    - functions for calculating the beam in sky/pixel coords
    - the ability to change the beam according to dec-lat
    """

    @classmethod
    def from_header(cls, header, beam=None, lat=None):
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
        return cls(wcs, beam, pixscale, refpix, lat)

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

    def sky2pix_ellipse(self, pos, a, b, pa):
        """
        Convert an ellipse from sky to pixel corrds
        a/b vectors are calculated at an origin pos=(ra,dec)
        All input parameters are in degrees
        Output parameters are:
        x,y - the x,y pixels corresponding to the ra/dec position
        sx, sy - the major minor axes (FWHM) in pixels
        theta - the position angle in degrees

        :param pos: [ra,dec] of the ellipse center
        :param a: major axis
        :param b: minor axis
        :param pa: position angle
        :return: x, y, sx, sy, theta
        """
        ra, dec = pos
        x, y = self.sky2pix(pos)

        x_off, y_off = self.sky2pix(translate(ra, dec, a, pa))
        sx = np.hypot((x - x_off),(y - y_off))
        theta = np.arctan2((y_off - y), (x_off - x))

        x_off, y_off = self.sky2pix(translate(ra, dec, b, pa-90))
        sy = np.hypot((x - x_off), (y - y_off))
        theta2 = np.arctan2((y_off - y), (x_off - x)) - np.pi/2

        # The a/b vectors are perpendicular in sky space, but not always in pixel space
        # so we have to account for this by calculating the angle between the two vectors
        # and modifying the minor axis length
        defect = theta - theta2
        sy *= abs(np.cos(defect))

        return x, y, sx, sy, np.degrees(theta)

    def pix2sky_ellipse(self, pixel, sx, sy, theta):
        """
        Convert an ellipse from pixel to sky coords
        sx/sy vectors are calculated at an origin pos=(x,y)
        Input parameters are:
        x,y - the x,y pixels corresponding to the ra/dec position
        sx, sy - the major minor axes (FWHM) in pixels
        theta - the position angle in degrees
        Output params are all in degrees

        :param pixel: [x,y] of the ellipse center
        :param sx: major axis
        :param sy: minor axis
        :param theta: position angle
        :return: ra, dec, a, b, pa
        """
        ra, dec = self.pix2sky(pixel)
        x, y = pixel
        v_sx = [x + sx * np.cos(np.radians(theta)),
                y + sx * np.sin(np.radians(theta))]
        ra2, dec2 = self.pix2sky(v_sx)
        major = gcd(ra, dec, ra2, dec2)
        pa = bear(ra, dec, ra2, dec2)

        v_sy = [x + sy * np.cos(np.radians(theta-90)),
                y + sy * np.sin(np.radians(theta-90))]
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
        A wrapper around get_pixbeam for when you only know the pixel coords
        :param x:
        :param y:
        :return:
        """
        ra, dec = self.pix2sky([x, y])
        return self.get_pixbeam(ra, dec)

    def get_beam(self, ra, dec):
        """
        Determine the beam at the given location
        The major axis of the beam is scaled by latitude if the lat is known.
        :param ra: Sky coord
        :param dec: Sky coord
        :return: Beam(a,b,pa)
        """
        # check to see if we need to scale the major axis based on the declination
        if self.lat is None:
            factor = 1
        else:
            # this works if the pa is zero. For non-zero pa it's a little more difficult
            factor = np.cos(np.radians(dec-self.lat))
        return Beam(self.beam.a/factor, self.beam.b, self.beam.pa)

    def get_pixbeam(self, ra, dec):
        """
        Use global_data to get beam (sky scale), and img.pixscale.
        Calculate a beam in pixel scale
        :return: A beam in pixel scale
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

        :param ra:
        :param dec:
        :return:
        """
        barea = abs(self.beam.a * self.beam.b * np.pi)  # in deg**2 at reference coords
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
        parea = abs(self.pixscale[0] * self.pixscale[1])  # in deg**2 at reference coords
        barea = self.get_beamarea_deg2(ra, dec)
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
        # self.helper = WCSHelper.from_file('Test/Images/1904-66_SIN.fits')
        self.helper = WCSHelper.from_file('Test/Week2_small.fits')
        self.test_vector_round_trip()
        self.test_ellipse_round_trip()
        self.test_defect()

    def test_vector_round_trip(self):
        """
        Converting a vector from pixel to sky coords and back again should give the
        original vector (within some tolerance).
        """
        print "Testing vector round trip... ",
        initial = [1, 45]  #r,theta = 1,45 (degrees)
        ref = self.helper.refpix
        ra, dec, dist, ang = self.helper.pix2sky_vec(ref, *initial)
        x, y, r, theta = self.helper.sky2pix_vec([ra, dec], dist, ang)
        print "Start: x {0}, y {1}, r {2}, theta {3}".format(ref[0], ref[1], *initial)
        print "sky: ra {0}, dec {1}, dist {2}, ang {3}".format(ra, dec, dist, ang)
        print "Final: x {0}, y {1}, r {2}, theta {3}".format(x, y, r, theta)
        if abs(r-initial[0]) < 1e-9 and abs(theta-initial[1]) < 1e-9:
            print "Pass"
            return True
        else:
            print "Fail"
            return False

    def test_ellipse_round_trip(self):
        """
        Converting an ellipse from pixel to sky coords and back again should give the
        original ellipse (within some tolerance).
        """
        print "Testing ellipse round trip"
        # raref, decref = self.helper.pix2sky(self.helper.refpix)
        a = 2*self.helper.beam.a
        b = self.helper.beam.b
        pa = self.helper.beam.pa+45
        ralist = range(-60, 181, 5)
        declist = range(-85, 86, 5)
        ras, decs = np.meshgrid(ralist, declist)
        # fmt = "RA: {0:5.2f} DEC: {1:5.2f} a: {2:5.2f} b: {3:5.2f} pa: {4:5.2f}"
        bgrid = np.empty(ras.shape[0]*ras.shape[1], dtype=np.float)
        for i, (ra, dec) in enumerate(zip(ras.ravel(), decs.ravel())):
            if ra < 0:
                ra += 360
            x, y, sx, sy, theta = self.helper.sky2pix_ellipse([ra, dec], a, b, pa)
            final = self.helper.pix2sky_ellipse([x, y], sx, sy, theta)
            bgrid[i] = final[3]
        bgrid = np.log(bgrid.reshape(ras.shape)/b)

        from matplotlib import pyplot
        figure = pyplot.figure()
        ax = figure.add_subplot(111)
        mappable = ax.imshow(bgrid, interpolation='nearest')
        cax = pyplot.colorbar(mappable)
        pyplot.show()


    def test_defect(self):
        """
        Make a plot showing the defect that occurs when converting major/minor axes
        from sky->pix coordinates
        """
        print "Testing defect"
        # raref, decref = self.helper.pix2sky(self.helper.refpix)
        a = 2*self.helper.beam.a
        b = self.helper.beam.b
        pa = self.helper.beam.pa+45
        ralist = range(-60,181,5)
        declist = range(-85,86,5)
        ras, decs = np.meshgrid(ralist,declist)
        # fmt = "RA: {0:5.2f} DEC: {1:5.2f} a: {2:5.2f} b: {3:5.2f} pa: {4:5.2f}"
        bgrid = np.empty(ras.shape[0]*ras.shape[1],dtype=np.float)
        for i,(ra, dec) in enumerate(zip(ras.ravel(),decs.ravel())):
            if ra<0:
                ra+=360
            initial = (ra,dec,a,b,pa)
            x,y,sx,theta = self.helper.sky2pix_vec([ra,dec],a,pa)
            _, _, sy, theta2 = self.helper.sky2pix_vec([ra,dec],b,pa+90)
            #final = self.helper.pix2sky_ellipse([x,y],sx,sy,theta)
            defect = theta-theta2-90
            bgrid[i] = 1/np.cos(np.radians(defect))
            # print '-'
            # print fmt.format(*initial),"->"
            # print fmt.format(*final)
        bgrid = bgrid.reshape(ras.shape)

        from matplotlib import pyplot
        figure = pyplot.figure()
        ax = figure.add_subplot(111)
        mappable = ax.imshow(bgrid, interpolation='nearest')
        cax = pyplot.colorbar(mappable)
        pyplot.show()


class PSFHelper(WCSHelper):
    """
    A class that will store information about the PSF, which is assumed to be direction dependent.
    PSFHelper contains a WCSHelper within, providing an extra layer of functionality including:
    - the ability to load psf/beam information from a fits file
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
        Determine the local psf (a,b,pa) at a given sky location.
        The beam is in sky coords.
        :param ra:
        :param dec:
        :return: (a,b,pa) in degrees
        """

        # If we don't have a psf map then we just fall back to using the beam
        # from the fits header (including ZA scaling)
        if self.data is None:
            beam = self.wcshelper.get_beam(ra,dec)
            return beam.a, beam.b, beam.pa
        
        x, y = self.sky2pix([ra, dec])                
        # We leave the interpolation in the hands of whoever is making these images
        # clamping the x,y coords at the image boundaries just makes sense
        x = int(np.clip(x, 0, self.data.shape[1]-1))
        y = int(np.clip(y, 0, self.data.shape[2]-1))
        psf_sky = self.data[:, x, y]
        return psf_sky

    def get_psf_pix(self, ra, dec):
        """
        Determine the local psf (a,b,pa) at a given sky location.
        The beam is in pixel coords.
        :param ra:
        :param dec:
        :return: (a,b,) in pix,pix,degrees
        """
        psf_sky = self.get_psf_sky(ra, dec)
        psf_pix = self.wcshelper.sky2pix_ellipse([ra, dec], psf_sky[0], psf_sky[1], psf_sky[2])[2:]
        # psf_pix = self.wcshelper.sky2pix_vec([ra, dec], psf_sky[0], 0)[2],\
        #           self.wcshelper.sky2pix_vec([ra, dec], psf_sky[1], 90)[2],\
        #           psf_sky[2]
        return psf_pix

    def get_pixbeam_pixel(self, x, y):
        """
        Get the beam shape at the location specified by pixel coords.
        :param x: pixel coord
        :param y: pixel coord
        :return: Beam(a,b,pa)
        """
        # overriding the WCSHelper function of the same name means that we now calculate the
        # psf at the coordinates of the x/y pixel in the image WCS, rather than the psfimage WCS
        ra, dec = self.wcshelper.pix2sky([x, y])
        return self.get_pixbeam(ra, dec)

    def get_pixbeam(self, ra, dec):
        """
        Get the beam at this location.
        :param ra: Sky coord
        :param dec: Sky coord
        :return: Beam(a,b,pa)
        """
        beam = self.wcshelper.get_pixbeam(ra, dec)
        # If there is no psf image then just use the fits header (plus lat scaling) from the wcshelper
        if self.data is None:
            return beam
        # get the beam from the psf image data
        psf = self.get_psf_pix(ra, dec)
        if None in psf:
            log.warn("PSF requested, returned Null")
            return None
        if np.isfinite(psf[0]):
            beam = Beam(psf[0], psf[1], psf[2])
        return beam

    def get_beam(self, ra, dec):
        """
        """
        if self.data is None:
            return self.wcshelper.beam
        else:
            psf = self.get_psf_sky(ra, dec)
            if not all(np.isfinite(psf)):
                return None
            return Beam(psf[0], psf[1], psf[2])

    def get_beamarea_pix(self, ra, dec):
        beam = self.get_pixbeam(ra, dec)
        if beam is None:
            return 0
        return beam.a*beam.b*np.pi

    def get_beamarea_deg2(self, ra, dec):
        """

        :param ra:
        :param dec:
        :return:
        """
        beam = self.get_beam(ra, dec)
        if beam is None:
            return 0
        return beam.a*beam.b*np.pi


class PSFHelperTest(object):
    """

    """
    def __init__(self):
        psffile = "Test/Images/1904_66_SIN_psf.fits"
        wcsfile = "Test/Images/1904_66_SIN.fits"
        psfdata = fits.getdata(psffile)
        self.helper = PSFHelper(psfdata,WCSHelper.from_file(wcsfile))
