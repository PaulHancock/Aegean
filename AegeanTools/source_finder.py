#! /usr/bin/env python
"""
The Aegean source finding program.
"""

# standard imports
import sys
import os
import re
import numpy as np
import math
import copy

# TODO: Not all of these modules are needed for every instance of Aegean.
# see if there is a way to only import the things that I need.
# this should increase load speed, and reduce complaints from modules that are not being used.

# scipy things
import scipy
from scipy import ndimage as ndi
from scipy.special import erf
from scipy.ndimage import label, find_objects

# fitting
import lmfit
from AegeanTools.fitting import do_lmfit, Cmatrix, Bmatrix, errors, covar_errors
from AegeanTools.fitting import ntwodgaussian_lmfit

# the glory of astropy
import astropy

# need Region in the name space in order to be able to unpickle it
try:
    from AegeanTools.regions import Region

    region_available = True
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
except ImportError:
    region_available = False

# logging import and setupOB
import logging
import logging.config

# command line option handler
from optparse import OptionParser

# external and support programs
from AegeanTools.wcs_helpers import WCSHelper, PSFHelper
from AegeanTools.fits_image import FitsImage, Beam
from AegeanTools.msq2 import MarchingSquares
from AegeanTools.angle_tools import dec2hms, dec2dms, gcd, bear, translate
import AegeanTools.flags as flags
from AegeanTools.catalogs import show_formats, check_table_formats, load_table, \
                                 load_catalog, table_to_source_list, save_catalog
from AegeanTools.models import OutputSource, IslandSource, SimpleSource, classify_catalog

# multiple cores support
import AegeanTools.pprocess as pprocess
import multiprocessing

header = """#Aegean version {0}
# on dataset: {1}"""

log = None
# global constants
FWHM2CC = 1 / (2 * math.sqrt(2 * math.log(2)))
CC2FHWM = (2 * math.sqrt(2 * math.log(2)))


class GlobalFittingData(object):
    """
    The global data used for fitting.
    (should be) Read-only once created.
    Used by island fitting subprocesses.
    """

    def __init__(self):
        self.img = None
        self.dcurve = None
        self.rmsimg = None
        self.bkgimg = None
        self.hdu_header = None
        self.beam = None
        self.data_pix = None
        self.dtype = None
        self.region = None
        self.wcshelper = None
        self.psfhelper = None
        return


class IslandFittingData(object):
    """
    All the data required to fit a single island.
    Instances are pickled and passed to the fitting subprocesses

    isle_num = island number (int)
    i = the pixel island (a 2D numpy array of pixel values)
    scalars=(innerclip,outerclip,max_summits)
    offsets=(xmin,xmax,ymin,ymax)
    """

    def __init__(self, isle_num=0, i=None, scalars=None, offsets=(0,0,1,1), doislandflux=False):
        self.isle_num = isle_num
        self.i = i
        self.scalars = scalars
        self.offsets = offsets
        self.doislandflux = doislandflux


class DummyLM(object):
    """
    A dummy copy of the lmfit results, for use when no fitting was done.
    """

    def __init__(self):
        self.residual = [np.nan,np.nan]
        self.success = False


class SourceFinder(object):
    """
    The Aegean source finding program
    """

    def __init__(self):
        self.globaldata = GlobalFittingData()
        self.globaldata.img = None
        self.globaldata.dcurve = None
        self.globaldata.rmsimg = None
        self.globaldata.bkgimg = None
        self.globaldata.hdu_header = None
        self.globaldata.beam = None
        self.globaldata.data_pix = None
        self.globaldata.dtype = None
        self.globaldata.region = None
        self.globaldata.wcshelper = None
        self.globaldata.psfhelper = None
        return

    def estimate_lmfit_parinfo(self, data, rmsimg, curve, beam, innerclip, outerclip=None, offsets=(0, 0), max_summits=None):
        """
        Estimates the number of sources in an island and returns initial parameters for the fit as well as
        limits on those parameters.

        :param data: np.ndarray of flux values
        :param rmsimg: np.ndarray of 1sigma values
        :param curve: np.ndarray of curvature values
        :param beam: beam object
        :param innerclip: the inner clipping level for flux data, in sigmas
        :param outerclip: the outer clipping level for flux data, in sigmas
        :param offsets: the (x,y) offset of data within it's parent image
        :param max_summits: if not None, only this many summits/components will be fit. More components may be
                            present in the island, but subsequent components will not have free parameters
        :return: an lmfit.Parameters object that describes our model
        """

        debug_on = log.isEnabledFor(logging.DEBUG)
        is_flag = 0

        # check to see if this island is a negative peak since we need to treat such cases slightly differently
        isnegative = max(data[np.where(np.isfinite(data))]) < 0
        if isnegative:
            log.debug("[is a negative island]")

        if outerclip is None:
            outerclip = innerclip

        log.debug(" - shape {0}".format(data.shape))

        if not data.shape == curve.shape:
            log.error("data and curvature are mismatched")
            log.error("data:{0} curve:{1}".format(data.shape, curve.shape))
            raise AssertionError()

        # For small islands we can't do a 6 param fit
        # Don't count the NaN values as part of the island
        non_nan_pix = len(data[np.where(np.isfinite(data))].ravel())
        if 4 <= non_nan_pix <= 6:
            log.debug("FIXED2PSF")
            is_flag |= flags.FIXED2PSF
        elif non_nan_pix < 4:
            log.debug("FITERRSMALL!")
            is_flag |= flags.FITERRSMALL
        else:
            is_flag = 0
        if debug_on:
            log.debug(" - size {0}".format(len(data.ravel())))

        if min(data.shape) <= 2 or (is_flag & flags.FITERRSMALL) or (is_flag & flags.FIXED2PSF):
            # 1d islands or small islands only get one source
            if debug_on:
                log.debug("Tiny summit detected")
                log.debug("{0}".format(data))
            summits = [[data, 0, data.shape[0], 0, data.shape[1]]]
            # and are constrained to be point sources
            is_flag |= flags.FIXED2PSF
        else:
            if isnegative:
                # the summit should be able to include all pixels within the island not just those above innerclip
                kappa_sigma = np.where(curve > 0.5, np.where(data + outerclip * rmsimg < 0, data, np.nan), np.nan)
            else:
                kappa_sigma = np.where(-1 * curve > 0.5, np.where(data - outerclip * rmsimg > 0, data, np.nan), np.nan)
            summits = list(gen_flood_wrap(kappa_sigma, np.ones(kappa_sigma.shape), 0, domask=False))

        params = lmfit.Parameters()
        i = 0
        summits_considered = 0
        # This can happen when the image contains regions of nans
        # the data/noise indicate an island, but the curvature doesn't back it up.
        if len(summits)<1:
            log.debug("Island has {0} summits".format(len(summits)))
            return None

        # add summits in reverse order of peak SNR - ie brightest first
        for summit, xmin, xmax, ymin, ymax in sorted(summits, key=lambda x: np.nanmax(-1.*abs(x[0]))):
            summits_considered += 1
            summit_flag = is_flag
            if debug_on:
                log.debug("Summit({5}) - shape:{0} x:[{1}-{2}] y:[{3}-{4}]".format(summit.shape, ymin, ymax, xmin, xmax, i))
            try:
                if isnegative:
                    amp = np.nanmin(summit)
                    xpeak, ypeak = np.unravel_index(np.nanargmin(summit), summit.shape)
                else:
                    amp = np.nanmax(summit)
                    xpeak, ypeak = np.unravel_index(np.nanargmax(summit), summit.shape)
            except ValueError, e:
                if "All-NaN" in e.message:
                    log.warn("Summit of nan's detected - this shouldn't happen")
                    continue
                else:
                    raise e

            if debug_on:
                log.debug(" - max is {0:f}".format(amp))
                log.debug(" - peak at {0},{1}".format(xpeak, ypeak))
            yo = ypeak + ymin
            xo = xpeak + xmin

            # Summits are allowed to include pixels that are between the outer and inner clip
            # This means that sometimes we get a summit that has all it's pixels below the inner clip
            # So we test for that here.
            snr = np.nanmax(abs(data[xmin:xmax+1, ymin:ymax+1] / rmsimg[xmin:xmax+1, ymin:ymax+1]))
            if snr < innerclip:
                log.debug("Summit has SNR {0} < innerclip {1}: skipping".format(snr, innerclip))
                continue


            # allow amp to be 5% or (innerclip) sigma higher
            # TODO: the 5% should depend on the beam sampling
            # note: when innerclip is 400 this becomes rather stupid
            if amp > 0:
                amp_min, amp_max = 0.95 * min(outerclip * rmsimg[xo, yo], amp), amp * 1.05 + innerclip * rmsimg[xo, yo]
            else:
                amp_max, amp_min = 0.95 * max(-outerclip * rmsimg[xo, yo], amp), amp * 1.05 - innerclip * rmsimg[xo, yo]

            if debug_on:
                log.debug("a_min {0}, a_max {1}".format(amp_min, amp_max))


            pixbeam = global_data.psfhelper.get_pixbeam_pixel(yo+offsets[0], xo+offsets[1])
            if pixbeam is None:
                log.debug(" Summit has invalid WCS/Beam - Skipping.".format(i))
                continue

            # set a square limit based on the size of the pixbeam
            xo_lim = 0.5*np.hypot(pixbeam.a, pixbeam.b)
            yo_lim = xo_lim

            yo_min, yo_max = yo - yo_lim, yo + yo_lim
            #if yo_min == yo_max:  # if we have a 1d summit then allow the position to vary by +/-0.5pix
            #    yo_min, yo_max = yo_min - 0.5, yo_max + 0.5

            xo_min, xo_max = xo - xo_lim, xo + xo_lim
            #if xo_min == xo_max:  # if we have a 1d summit then allow the position to vary by +/-0.5pix
            #    xo_min, xo_max = xo_min - 0.5, xo_max + 0.5

            # the size of the island
            xsize = data.shape[0]
            ysize = data.shape[1]

            # initial shape is the psf
            sx = pixbeam.a * FWHM2CC
            sy = pixbeam.b * FWHM2CC

            # lmfit does silly things if we start with these two parameters being equal
            sx = max(sx, sy*1.01)

            # constraints are based on the shape of the island
            sx_min, sx_max = sx * 0.8, max((max(xsize, ysize) + 1) * math.sqrt(2) * FWHM2CC, sx * 1.1)
            sy_min, sy_max = sy * 0.8, max((max(xsize, ysize) + 1) * math.sqrt(2) * FWHM2CC, sx * 1.1)

            theta = pixbeam.pa # Degrees
            flag = summit_flag

            # check to see if we are going to fit this component
            if max_summits is not None:
                maxxed = i >= max_summits
            else:
                maxxed = False

            # components that are not fit need appropriate flags
            if maxxed:
                summit_flag |= flags.NOTFIT
                summit_flag |= flags.FIXED2PSF

            if debug_on:
                log.debug(" - var val min max | min max")
                log.debug(" - amp {0} {1} {2} ".format(amp, amp_min, amp_max))
                log.debug(" - xo {0} {1} {2} ".format(xo, xo_min, xo_max))
                log.debug(" - yo {0} {1} {2} ".format(yo, yo_min, yo_max))
                log.debug(" - sx {0} {1} {2} | {3} {4}".format(sx, sx_min, sx_max, sx_min * CC2FHWM,
                                                                      sx_max * CC2FHWM))
                log.debug(" - sy {0} {1} {2} | {3} {4}".format(sy, sy_min, sy_max, sy_min * CC2FHWM,
                                                                      sy_max * CC2FHWM))
                log.debug(" - theta {0} {1} {2}".format(theta, -180, 180))
                log.debug(" - flags {0}".format(flag))
                log.debug(" - fit?  {0}".format(not maxxed))

            # TODO: figure out how incorporate the circular constraint on xo/yo
            prefix = "c{0}_".format(i)
            params.add(prefix+'amp',value=amp, min=amp_min, max=amp_max, vary= not maxxed)
            params.add(prefix+'xo',value=xo, min=float(xo_min), max=float(xo_max), vary= not maxxed)
            params.add(prefix+'yo',value=yo, min=float(yo_min), max=float(yo_max), vary= not maxxed)

            if summit_flag & flags.FIXED2PSF > 0:
                psf_vary = False
            else:
                psf_vary = not maxxed
            params.add(prefix+'sx', value=sx, min=sx_min, max=sx_max, vary=psf_vary)
            params.add(prefix+'sy', value=sy, min=sy_min, max=sy_max, vary=psf_vary)
            params.add(prefix+'theta', value=theta, vary=psf_vary)
            params.add(prefix+'flags', value=summit_flag, vary=False)

            # starting at zero allows the maj/min axes to be fit better.
            if params[prefix+'theta'].vary:
                params[prefix+'theta'].value = 0

            i += 1
        if debug_on:
            log.debug("Estimated sources: {0}".format(i))
        # remember how many components are fit.
        params.add('components',value=i, vary=False)
        #params.components=i
        if params['components'].value <1:
            log.debug("Considered {0} summits, accepted {1}".format(summits_considered,i))
        return params

    def result_to_components(self, result, model, island_data, isflags):
        """
        Convert fitting results into a set of components

        :param result: the results from lmfit (pixel data etc.)
        :param model: the model that was fit
        :param island_data: an IslandFittingData object
        :param isflags: flags that should be added to this island (in addition to those within the model)
        :return: a list of components [and islands]
        """

        global_data = self.globaldata

        # island data
        isle_num = island_data.isle_num
        idata = island_data.i
        xmin, xmax, ymin, ymax = island_data.offsets

        rms = global_data.rmsimg[xmin:xmax, ymin:ymax]
        bkg = global_data.bkgimg[xmin:xmax, ymin:ymax]
        residual = np.median(result.residual), np.std(result.residual)
        is_flag = isflags

        sources = []
        for j in range(model['components'].value):
            src_flags = is_flag
            source = OutputSource()
            source.island = isle_num
            source.source = j
            log.debug(" component {0}".format(j))
            prefix = "c{0}_".format(j)
            xo = model[prefix+'xo'].value
            yo = model[prefix+'yo'].value
            sx = model[prefix+'sx'].value
            sy = model[prefix+'sy'].value
            theta = model[prefix+'theta'].value
            amp = model[prefix+'amp'].value
            src_flags |= model[prefix+'flags'].value

            # these are goodness of fit statistics for the entire island.
            source.residual_mean = residual[0]
            source.residual_std = residual[1]
            # set the flags
            source.flags = src_flags

            # #pixel pos within island +
            # island offset within region +
            # region offset within image +
            # 1 for luck
            # (pyfits->fits conversion = luck)
            x_pix = xo + xmin + 1
            y_pix = yo + ymin + 1
            # update the source xo/yo so the error calculations can be done correctly
            # Note that you have to update the max or the value you set will be clipped at the max allowed value
            model[prefix+'xo'].set(value=x_pix, max=np.inf)
            model[prefix+'yo'].set(value=y_pix, max=np.inf)
            # ------ extract source parameters ------

            # fluxes
            # the background is taken from background map
            # Clamp the pixel location to the edge of the background map
            y = max(min(int(round(y_pix - ymin)), bkg.shape[1] - 1), 0)
            x = max(min(int(round(x_pix - xmin)), bkg.shape[0] - 1), 0)
            source.background = bkg[x, y]
            source.local_rms = rms[x, y]
            source.peak_flux = amp

            # all params are in degrees
            source.ra, source.dec, source.a, source.b, source.pa = global_data.wcshelper.pix2sky_ellipse((x_pix, y_pix), sx*CC2FHWM, sy*CC2FHWM, theta)
            source.a *= 3600  # arcseconds
            source.b *= 3600
            # force a>=b
            fix_shape(source)
            # limit the pa to be in (-90,90]
            source.pa = pa_limit(source.pa)

            # if one of these values are nan then there has been some problem with the WCS handling
            if not all(np.isfinite((source.ra, source.dec, source.a, source.b, source.pa))):
                src_flags |= flags.WCSERR
            # negative degrees is valid for RA, but I don't want them.
            if source.ra < 0:
                source.ra += 360
            source.ra_str = dec2hms(source.ra)
            source.dec_str = dec2dms(source.dec)



            # calculate integrated flux
            source.int_flux = source.peak_flux * sx * sy * CC2FHWM ** 2 * np.pi
            # scale Jy/beam -> Jy using the area of the beam
            source.int_flux /= global_data.psfhelper.get_beamarea_pix(source.ra, source.dec)

            # Calculate errors for params that were fit (as well as int_flux)
            errors(source, model, global_data.wcshelper)

            source.flags = src_flags
            # add psf info
            local_beam = global_data.psfhelper.get_beam(source.ra, source.dec)
            if local_beam is not None:
                source.psf_a = local_beam.a*3600
                source.psf_b = local_beam.b*3600
                source.psf_pa = local_beam.pa
            else:
                source.psf_a = 0
                source.psf_b = 0
                source.psf_pa = 0
            sources.append(source)
            log.debug(source)

        # calculate the integrated island flux if required
        if island_data.doislandflux:
            _, outerclip, _ = island_data.scalars
            log.debug("Integrated flux for island {0}".format(isle_num))
            kappa_sigma = np.where(abs(idata) - outerclip * rms > 0, idata, np.NaN)
            log.debug("- island shape is {0}".format(kappa_sigma.shape))

            source = IslandSource()
            source.flags = 0
            source.island = isle_num
            source.components = j + 1
            source.peak_flux = np.nanmax(kappa_sigma)
            # check for negative islands
            if source.peak_flux < 0:
                source.peak_flux = np.nanmin(kappa_sigma)
            log.debug("- peak flux {0}".format(source.peak_flux))

            # positions and background
            if np.isfinite(source.peak_flux):
                positions = np.where(kappa_sigma == source.peak_flux)
            else: # if a component has been refit then it might have flux = np.nan
                positions = [[kappa_sigma.shape[0]/2],[ kappa_sigma.shape[1]/2]]
            xy = positions[0][0] + xmin, positions[1][0] + ymin
            radec = global_data.wcshelper.pix2sky(xy)
            source.ra = radec[0]

            # convert negative ra's to positive ones
            if source.ra < 0:
                source.ra += 360

            source.dec = radec[1]
            source.ra_str = dec2hms(source.ra)
            source.dec_str = dec2dms(source.dec)
            source.background = bkg[positions[0][0], positions[1][0]]
            source.local_rms = rms[positions[0][0], positions[1][0]]
            source.x_width, source.y_width = idata.shape
            source.pixels = int(sum(np.isfinite(kappa_sigma).ravel() * 1.0))
            source.extent = [xmin, xmax, ymin, ymax]

            # TODO: investigate what happens when the sky coords are skewed w.r.t the pixel coords
            # calculate the area of the island as a fraction of the area of the bounding box
            bl = global_data.wcshelper.pix2sky([xmax, ymin])
            tl = global_data.wcshelper.pix2sky([xmax, ymax])
            tr = global_data.wcshelper.pix2sky([xmin, ymax])
            height = gcd(tl[0], tl[1], bl[0], bl[1])
            width = gcd(tl[0], tl[1], tr[0], tr[1])
            area = height * width
            source.area = area * source.pixels / source.x_width / source.y_width  # area is in deg^2

            # create contours
            msq = MarchingSquares(idata)
            source.contour = [(a[0] + xmin, a[1] + ymin) for a in msq.perimeter]
            # calculate the maximum angular size of this island, brute force method
            source.max_angular_size = 0
            for i, pos1 in enumerate(source.contour):
                radec1 = global_data.wcshelper.pix2sky(pos1)
                for j, pos2 in enumerate(source.contour[i:]):
                    radec2 = global_data.wcshelper.pix2sky(pos2)
                    dist = gcd(radec1[0], radec1[1], radec2[0], radec2[1])
                    if dist > source.max_angular_size:
                        source.max_angular_size = dist
                        source.pa = bear(radec1[0], radec1[1], radec2[0], radec2[1])
                        source.max_angular_size_anchors = [pos1[0], pos1[1], pos2[0], pos2[1]]

            log.debug("- peak position {0}, {1} [{2},{3}]".format(source.ra_str, source.dec_str, positions[0][0],
                                                                      positions[1][0]))

            # integrated flux
            beam_area = global_data.psfhelper.get_beamarea_deg2(source.ra, source.dec)  # beam in deg^2
            # get_beamarea_pix(source.ra, source.dec)  # beam is in pix^2
            isize = source.pixels  # number of non zero pixels
            log.debug("- pixels used {0}".format(isize))
            source.int_flux = np.nansum(kappa_sigma)  # total flux Jy/beam
            log.debug("- sum of pixles {0}".format(source.int_flux))
            source.int_flux *= beam_area  # total flux in Jy
            log.debug("- integrated flux {0}".format(source.int_flux))
            eta = erf(np.sqrt(-1 * np.log(abs(source.local_rms * outerclip / source.peak_flux)))) ** 2
            log.debug("- eta {0}".format(eta))
            source.eta = eta
            source.beam_area = beam_area

            # I don't know how to calculate this error so we'll set it to nan
            source.err_int_flux = np.nan
            sources.append(source)
        return sources


# Helpers
def fix_shape(source):
    """
    Ensure that a>=b for a given source object
    if a<b then swap a/b and increment pa by 90
    err_a/err_b are also swapped as needed
    :param source: any object with a/b/pa/err_a/err_b properties
    :return: None
    """
    if source.a < source.b:
        source.a, source.b = source.b, source.a
        source.err_a, source.err_b = source.err_b, source.err_a
        source.pa += 90
    return

def pa_limit(pa):
    """
    Position angle is periodic with period 180\deg
    Constrain pa such that -90<pa<=90

    :param pa: position angle (degrees)
    :return: position angle (degrees)
    """
    while pa <= -90:
        pa += 180
    while pa > 90:
        pa -= 180
    return pa

def theta_limit(theta):
    """
    Angle theta is periodic with period pi
    Constrain theta such that -pi/2<theta<=pi/2
    :param theta: angle in radians
    :return: angle in radians
    """
    while theta <= -1*np.pi/2:
        theta += np.pi
    while theta > np.pi/2:
        theta -= np.pi
    return theta

def scope2lat(telescope):
    """
    Convert a telescope name into a latitude
    returns None when the telescope is unknown.

    :param telescope: a string
    :return: a latitude or None
    """
    # Note: these values were taken from wikipedia so have varying precision/accuracy
    scopes = {'MWA':-26.703319,
              "ATCA":-30.3128,
              "VLA":34.0790,
              "LOFAR":52.9088,
              "KAT7":-30.721,
              "MEERKAT":-30.721,
              "PAPER":-30.7224,
              "GMRT":19.096516666667,
              "OOTY":11.383404,
              "ASKAP":-26.7,
              "MOST":-35.3707,
              "PARKES":-32.999944,
              "WSRT":52.914722,
              "AMILA":52.16977,
              "AMISA":52.164303,
              "ATA":40.817,
              "CHIME":49.321,
              "CARMA":37.28044,
              "DRAO":49.321,
              "GBT":38.433056,
              "LWA":34.07,
              "ALMA":-23.019283,
              "FAST":25.6525
              }
    if telescope.upper() in scopes:
        return scopes[telescope.upper()]
    else:
        log.warn("Telescope {0} is unknown".format(telescope))
        log.warn("integrated fluxes may be incorrect")
        return None

if __name__ == "__main__":
    print "this should do some testing"
    print "but it doesn't (yet)"
    sys.exit(0)