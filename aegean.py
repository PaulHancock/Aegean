#! /usr/bin/env python
"""
The Aegean source finding program.

Created by:
Paul Hancock
May 13 2011

Modifications by:
Paul Hancock
Jay Banyer
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

__author__ = 'Paul Hancock'

# Aegean version [Updated via script]
__version__ = '1.9.7-6-gde391c2'
__date__ = '2016-03-01'

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


def gen_flood_wrap(data, rmsimg, innerclip, outerclip=None, domask=False):
    """
    Generator function.
    Segment an image into islands and return one island at a time.

    Needs to work for entire image, and also for components within an island.

    :param data: and island not a 2d array of pixel values
    :param rmsimg: 2d array of rms values
    :param innerclip: seed clip value
    :param outerclip: flood clip value
    :param domask: look for a region mask in globals, and only return islands that are within the mask
    :return:
    """

    if outerclip is None:
        outerclip = innerclip

    # compute SNR image
    snr = abs(data)/rmsimg
    # mask of pixles that are above the outerclip
    a =  snr >= outerclip
    # segmentation a la scipy
    l, n = label(a)
    f = find_objects(l)

    if n == 0:
        log.debug("There are no pixels above the clipping limit")
        return

    # Yield values as before, though they are not sorted by flux
    for i in range(n):
        xmin,xmax = f[i][0].start, f[i][0].stop
        ymin,ymax = f[i][1].start, f[i][1].stop
        if np.any(snr[xmin:xmax,ymin:ymax]>innerclip): # obey inner clip constraint
            data_box = copy.copy(data[xmin:xmax,ymin:ymax]) # copy so that we don't blank the master data
            data_box[np.where(snr[xmin:xmax,ymin:ymax] < outerclip)] = np.nan # blank pixels that are outside the outerclip
            data_box[np.where(l[xmin:xmax,ymin:ymax] != i+1)] = np.nan        # blank out other summits
            # check if there are any pixels left unmasked
            if not np.any(np.isfinite(data_box)):
                continue
            if domask and global_data.region is not None:
                y,x = np.where(snr[xmin:xmax,ymin:ymax] >= outerclip)
                # convert indices of this sub region to indices in the greater image
                yx = zip(y+ymin,x+xmin)
                ra, dec = global_data.wcshelper.wcs.wcs_pix2world(yx, 1).transpose()
                mask = global_data.region.sky_within(ra, dec, degin=True)
                # if there are no un-masked pixels within the region then we skip this island.
                if not np.any(mask):
                    continue
                log.debug("Mask {0}".format(mask))
            yield data_box, xmin, xmax, ymin, ymax


# parameter estimates
def estimate_lmfit_parinfo(data, rmsimg, curve, beam, innerclip, outerclip=None, offsets=(0, 0), max_summits=None):
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
        sx = max(sx,sy*1.01)

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
        params.add(prefix+'flags',value=summit_flag, vary=False)

        i += 1
    if debug_on:
        log.debug("Estimated sources: {0}".format(i))
    # remember how many components are fit.
    params.add('components',value=i, vary=False)
    #params.components=i
    if params['components'].value <1:
        log.debug("Considered {0} summits, accepted {1}".format(summits_considered,i))
    return params


def result_to_components(result, model, island_data, isflags):
    """
    Convert fitting results into a set of components

    :param result: the results from lmfit (pixel data etc.)
    :param model: the model that was fit
    :param island_data: an IslandFittingData object
    :param isflags: flags that should be added to this island (in addition to those within the model)
    :return: a list of components [and islands]
    """

    global global_data

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
        source.int_flux /= global_data.wcshelper.get_beamarea_pix(source.ra, source.dec)

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


# load and save external files
def load_aux_image(image, auxfile):
    """
    Load a fits file (bkg/rms/curve) and make sure that
    it is the same shape as the main image.
    :param image: main FitsImage object
    :param auxfile: filename of auxiliary file to be loaded
    :return: FitsImage(auxfile)
    """
    auximg = FitsImage(auxfile, beam=global_data.beam).get_pixels()
    if auximg.shape != image.get_pixels().shape:
        log.error("file {0} is not the same size as the image map".format(auxfile))
        log.error("{0}= {1}, image = {2}".format(auxfile, auximg.shape, image.get_pixels().shape))
        sys.exit(1)
    return auximg


def load_globals(filename, hdu_index=0, bkgin=None, rmsin=None, beam=None, verb=False, rms=None, cores=1, csigma=None,
                 do_curve=True, mask=None, lat=None, psf=None):
    """
    Populate the global_data object by loading or calculating the various components

    :param filename: Main image which source finding is run on
    :param hdu_index: HDU index of the image within the fits file, default is 0 (first)
    :param bkgin: background image filename or HDUList
    :param rmsin: rms/noise image filename or HDUList
    :param beam: Beam object representing the synthsized beam. Will replace what is in the FITS header.
    :param verb: write extra lines to INFO level log
    :param rms: a float that represents a constant rms level for the entire image
    :param cores: number of cores to use if different from what is autodetected
    :param csigma: float value that represents the 1sigma value for the curvature map (don't use please)
    :param do_curve: if True a curvature map will be created, default=True
    :param mask: filename or Region object
    :param lat: latitude of the observing telescope (declination of zenith)
    :param psf: filename or HDUList of a psf image
    :return: None
    """
    global global_data

    img = FitsImage(filename, hdu_index=hdu_index, beam=beam)
    beam = img.beam

    # Save global data for use by fitting sub-processes
    global_data = GlobalFittingData()

    debug = logging.getLogger('Aegean').isEnabledFor(logging.DEBUG)

    if mask is None:
        global_data.region=None
    elif not region_available:
        log.warn("Mask supplied but functionality not available")
        global_data.region=None
    else:
        # allow users to supply and object instead of a filename
        if isinstance(mask, Region):
            global_data.region = mask
        elif os.path.exists(mask):
            log.info("Loading mask from {0}".format(mask))
            global_data.region = pickle.load(open(mask))
        else:
            log.error("File {0} not found for loading".format(mask))
            global_data.region=None

    global_data.wcshelper = WCSHelper.from_header(img.get_hdu_header(), beam, lat)
    global_data.psfhelper = PSFHelper(psf, global_data.wcshelper)

    global_data.beam = global_data.wcshelper.beam
    global_data.img = img
    global_data.data_pix = img.get_pixels()
    global_data.dtype = type(global_data.data_pix[0][0])
    global_data.bkgimg = np.zeros(global_data.data_pix.shape, dtype=global_data.dtype)
    global_data.rmsimg = np.zeros(global_data.data_pix.shape, dtype=global_data.dtype)
    global_data.pixarea = img.pixarea
    global_data.dcurve = None

    if do_curve:
        log.info("Calculating curvature")
        # calculate curvature but store it as -1,0,+1
        dcurve = np.zeros(global_data.data_pix.shape, dtype=np.int8)
        peaks = scipy.ndimage.filters.maximum_filter(global_data.data_pix, size=3)
        troughs = scipy.ndimage.filters.minimum_filter(global_data.data_pix, size=3)
        pmask = np.where(global_data.data_pix == peaks)
        tmask = np.where(global_data.data_pix == troughs)
        dcurve[pmask] = -1
        dcurve[tmask] = 1
        global_data.dcurve = dcurve

    # if either of rms or bkg images are not supplied then calculate them both
    if not (rmsin and bkgin):
        if verb:
            log.info("Calculating background and rms data")
        make_bkg_rms_from_global(mesh_size=20, forced_rms=rms, cores=cores)

    # if a forced rms was supplied use that instead
    if rms is not None:
        global_data.rmsimg = np.ones(global_data.data_pix.shape) * rms

    # replace the calculated images with input versions, if the user has supplied them.
    if bkgin:
        if verb:
            log.info("Loading background data from file {0}".format(bkgin))
        global_data.bkgimg = load_aux_image(img, bkgin)
    if rmsin:
        if verb:
            log.info("Loading rms data from file {0}".format(rmsin))
        global_data.rmsimg = load_aux_image(img, rmsin)

    # subtract the background image from the data image and save
    if verb and debug:
        log.debug("Data max is {0}".format(img.get_pixels()[np.isfinite(img.get_pixels())].max()))
        log.debug("Doing background subtraction")
    img.set_pixels(img.get_pixels() - global_data.bkgimg)
    global_data.data_pix = img.get_pixels()
    if verb and debug:
        log.debug("Data max is {0}".format(img.get_pixels()[np.isfinite(img.get_pixels())].max()))

    return

# image manipulation
def make_bkg_rms_image(data, beam, mesh_size=20, forced_rms=None):
    """
    [legacy version used by the VAST pipeline]

    Calculate an rms image and a bkg image

    inputs:
    data - np.ndarray of flux values
    beam - beam object
    mesh_size - number of beams per box
                default = 20
    forced_rms - the rms of the image
                None => calculate the rms and bkg levels (default)
                <float> => assume zero background and constant rms
    return:
    nothing

    """
    if forced_rms:
        return np.zeros(data.shape), np.ones(data.shape) * forced_rms

    img_x, img_y = data.shape
    xcen = int(img_x / 2)
    ycen = int(img_y / 2)

    #calculate a local beam from the center of the data
    #pixbeam = global_data.wcshelper.get_pixbeam()
    pixbeam = global_data.psfhelper.get_pixbeam_pixel(xcen,ycen)
    if pixbeam is None:
        log.error("Cannot calculate the beam shape at the image center")
        sys.exit(1)

    width_x = mesh_size * max(abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.a),
                              abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.b))
    width_x = int(width_x)
    width_y = mesh_size * max(abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.a),
                              abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.b))
    width_y = int(width_y)

    rmsimg = np.zeros(data.shape)
    bkgimg = np.zeros(data.shape)
    log.debug("image size x,y:{0},{1}".format(img_x, img_y))
    log.debug("beam: {0}".format(beam))
    log.debug("mesh width (pix) x,y: {0},{1}".format(width_x, width_y))

    #box centered at image center then tilling outwards
    xstart = (xcen - width_x / 2) % width_x  #the starting point of the first "full" box
    ystart = (ycen - width_y / 2) % width_y

    xend = img_x - (img_x - xstart) % width_x  #the end point of the last "full" box
    yend = img_y - (img_y - ystart) % width_y

    xmins = [0]
    xmins.extend(range(xstart, xend, width_x))
    xmins.append(xend)

    xmaxs = [xstart]
    xmaxs.extend(range(xstart + width_x, xend + 1, width_x))
    xmaxs.append(img_x)

    ymins = [0]
    ymins.extend(range(ystart, yend, width_y))
    ymins.append(yend)

    ymaxs = [ystart]
    ymaxs.extend(range(ystart + width_y, yend + 1, width_y))
    ymaxs.append(img_y)

    #if the image is smaller than our ideal mesh size, just use the whole image instead
    if width_x >= img_x:
        xmins = [0]
        xmaxs = [img_x]
    if width_y >= img_y:
        ymins = [0]
        ymaxs = [img_y]

    for xmin, xmax in zip(xmins, xmaxs):
        for ymin, ymax in zip(ymins, ymaxs):
            bkg, rms = estimate_bkg_rms(data[ymin:ymax, xmin:xmax])
            rmsimg[ymin:ymax, xmin:xmax] = rms
            bkgimg[ymin:ymax, xmin:xmax] = bkg

    return bkgimg, rmsimg


def make_bkg_rms_from_global(mesh_size=20, forced_rms=None, cores=None):
    """
    Calculate an rms image and a bkg image
    reads  data_pix, beam, rmsimg, bkgimg from global_data
    writes rmsimg, bkgimg to global_data
    is able to run on multiple cores

    :param mesh_size: number of beams per box default = 20
    :param forced_rms: the rms of the image
                       None => calculate the rms and bkg levels (default)
                       <float> => assume zero background and constant rms
    :param cores: number of cores to use, default = None = 1 core
    :return: None
    """
    if forced_rms:
        global_data.bkgimg[:] = 0
        global_data.rmsimg[:] = forced_rms
        return

    data = global_data.data_pix
    beam = global_data.beam

    img_x, img_y = data.shape
    xcen = int(img_x / 2)
    ycen = int(img_y / 2)

    # calculate a local beam from the center of the data
    pixbeam = global_data.psfhelper.get_pixbeam_pixel(xcen,ycen)
    if pixbeam is None:
        log.error("Cannot determine the beam shape at the image center")
        sys.exit(1)

    width_x = mesh_size * max(abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.a),
                              abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.b))
    width_x = int(width_x)
    width_y = mesh_size * max(abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.a),
                              abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.b))
    width_y = int(width_y)

    log.debug("image size x,y:{0},{1}".format(img_x, img_y))
    log.debug("beam: {0}".format(beam))
    log.debug("mesh width (pix) x,y: {0},{1}".format(width_x, width_y))

    # box centered at image center then tilling outwards
    xstart = (xcen - width_x / 2) % width_x  # the starting point of the first "full" box
    ystart = (ycen - width_y / 2) % width_y

    xend = img_x - (img_x - xstart) % width_x  #the end point of the last "full" box
    yend = img_y - (img_y - ystart) % width_y

    xmins = [0]
    xmins.extend(range(xstart, xend, width_x))
    xmins.append(xend)

    xmaxs = [xstart]
    xmaxs.extend(range(xstart + width_x, xend + 1, width_x))
    xmaxs.append(img_x)

    ymins = [0]
    ymins.extend(range(ystart, yend, width_y))
    ymins.append(yend)

    ymaxs = [ystart]
    ymaxs.extend(range(ystart + width_y, yend + 1, width_y))
    ymaxs.append(img_y)

    # if the image is smaller than our ideal mesh size, just use the whole image instead
    if width_x >= img_x:
        xmins = [0]
        xmaxs = [img_x]
    if width_y >= img_y:
        ymins = [0]
        ymaxs = [img_y]

    if cores > 1:
        # set up the queue
        queue = pprocess.Queue(limit=cores, reuse=1)
        estimate = queue.manage(pprocess.MakeReusable(estimate_background_global))
        # populate the queue
        for xmin, xmax in zip(xmins, xmaxs):
            for ymin, ymax in zip(ymins, ymaxs):
                estimate(ymin, ymax, xmin, xmax)
    else:
        queue = []
        for xmin, xmax in zip(xmins, xmaxs):
            for ymin, ymax in zip(ymins, ymaxs):
                queue.append(estimate_background_global(xmin, xmax, ymin, ymax))

    # construct the bkg and rms images
    if global_data.rmsimg is None:
        global_data.rmsimg = np.empty(data.shape, dtype=global_data.dtype)
    if global_data.bkgimg is None:
        global_data.bkgimg = np.empty(data.shape, dtype=global_data.dtype)

    for ymin, ymax, xmin, xmax, bkg, rms in queue:
        global_data.bkgimg[ymin:ymax, xmin:xmax] = bkg
        global_data.rmsimg[ymin:ymax, xmin:xmax] = rms
    return


def estimate_background_global(xmin, xmax, ymin, ymax):
    """
    Estimate the background noise mean and RMS.
    The mean is estimated as the median of data.
    The RMS is estimated as the IQR of data / 1.34896.

    reads/writes data from global_data
    works only on the sub-region specified by
    ymin,ymax,xmin,xmax
    Background and RMS are set to np.nan if data contains fewer than 4 non nan values.

    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :return: ymin, ymax, xmin, xmax, bkg, rms
    """
    data = global_data.data_pix[ymin:ymax, xmin:xmax]
    pixels = np.extract(np.isfinite(data), data).ravel()
    if len(pixels) < 4:
        bkg, rms = np.NaN, np.NaN
    else:
        pixels.sort()
        p25 = pixels[pixels.size / 4]
        p50 = pixels[pixels.size / 2]
        p75 = pixels[pixels.size / 4 * 3]
        iqr = p75 - p25
        bkg, rms = p50, iqr / 1.34896
    # return the input and output data so we know what we are doing
    # when compiling the results of multiple processes
    return ymin, ymax, xmin, xmax, bkg, rms


def estimate_bkg_rms(data):
    """
    Estimate the background noise mean and RMS.
    The mean is estimated as the median of data.
    The RMS is estimated as the IQR of data / 1.34896.

    :param data: image data
    :return: background, rms
    """
    pixels = np.extract(np.isfinite(data), data).ravel()
    if len(pixels) < 4:
        return np.NaN, np.NaN
    pixels.sort()
    p25 = pixels[pixels.size / 4]
    p50 = pixels[pixels.size / 2]
    p75 = pixels[pixels.size / 4 * 3]
    iqr = p75 - p25
    return p50, iqr / 1.34896


def curvature(data):
    """
    Use a Laplacian kernel to calculate the curvature of a 2d image.
    :param data: the image data
    :return: curvature image
    """
    kern = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return ndi.convolve(data, kern)


# Nifty helpers
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


def gmean(indata):
    """
    Calculate the geometric mean and variance of a data set.

    This function is designed such that gmean(data + a) = gmean(data) + a
    which means that it can operate on negative values

    np.nan values are excluded from the calculation however np.inf values cause the results to be also np.inf

    This is the function that you are looking for when asking, what is the mean/variance of the ratio x/y, or any other
    log-normal distributed data.

    :param data: an array of numbers
    :return: the geometric mean and variance of the data
    """
    # TODO: Figure out the mathematical name for functions that obey - gmean(data + a) = gmean(data) + a
    data = np.ravel(indata)
    if np.inf in data:
        return np.inf, np.inf

    finite = data[np.isfinite(data)]
    if len(finite) < 1:
        return np.nan, np.nan
    # determine the zero point and scale all values to be 1 or greater
    scale = abs(np.min(finite)) + 1
    finite += scale
    # calculate the geometric mean of the scaled data and scale back
    lfinite = np.log(finite)
    flux = np.exp(np.mean(lfinite)) - scale
    error = np.nanstd(lfinite) * flux
    return flux, abs(error)


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


######################################### THE MAIN DRIVING FUNCTIONS ###############

# source finding and fitting
def refit_islands(group, stage, outerclip, istart=0):
    """
    Do island refitting (priorized fitting) on a group of islands.

    :param group: A list of islands group=[ [(0,0),(0,1)],[(1,0)] ...]
    :param stage: refit stage
    :param outerclip: outerclip for floodfill. outerclip<0 means no clipping
    :param istart: the starting island number
    :return: a list of sources (including islands)
    """

    sources = []

    data = global_data.data_pix
    rmsimg = global_data.rmsimg

    for inum, isle in enumerate(group, start=istart):
        log.debug("-=-")
        log.debug("input island = {0}, {1} components".format(isle[0].island, len(isle)))

        # set up the parameters for each of the sources within the island
        i = 0
        params = lmfit.Parameters()
        shape = data.shape
        xmin, ymin = shape
        xmax = ymax = 0

        # island_mask = []
        src_valid_psf = None
        # keep track of the sources that are actually being refit
        # this may be a subset of all sources in the island
        included_sources = []
        for src in isle:
            pixbeam = global_data.psfhelper.get_pixbeam(src.ra, src.dec)
            # find the right pixels from the ra/dec
            source_x, source_y = global_data.wcshelper.sky2pix([src.ra, src.dec])
            source_x -= 1
            source_y -= 1
            x = int(round(source_x))
            y = int(round(source_y))

            log.debug("pixel location ({0:5.2f},{1:5.2f})".format(source_x, source_y))
            # reject sources that are outside the image bounds, or which have nan data/rms values
            if not 0 <= x < shape[0] or not 0 <= y < shape[1] or \
                    not np.isfinite(data[x, y]) or \
                    not np.isfinite(rmsimg[x, y]) or \
                    pixbeam is None:
                log.debug("Source ({0},{1}) not within usable region: skipping".format(src.island, src.source))
                continue
            else:
                # Keep track of the last source to have a valid psf so that we can use it later on
                src_valid_psf = src
            # determine the shape parameters in pixel values
            _, _, sx, sy, theta = global_data.wcshelper.sky2pix_ellipse([src.ra, src.dec], src.a/3600, src.b/3600, src.pa)
            sx *= FWHM2CC
            sy *= FWHM2CC

            log.debug("Source shape [sky coords]  {0:5.2f}x{1:5.2f}@{2:05.2f}".format(src.a, src.b, src.pa))
            log.debug("Source shape [pixel coords] {0:4.2f}x{1:4.2f}@{2:05.2f}".format(sx, sy, theta))

            # choose a region that is 2x the major axis of the source, 4x semimajor axis a
            width = 4 * sx
            ywidth = int(round(width)) + 1
            xwidth = int(round(width)) + 1

            # adjust the size of the island to include this source
            xmin = min(xmin, max(0, x - xwidth / 2))
            ymin = min(ymin, max(0, y - ywidth / 2))
            xmax = max(xmax, min(shape[0], x + xwidth / 2 + 1))
            ymax = max(ymax, min(shape[1], y + ywidth / 2 + 1))

            s_lims = [0.8 * min(sx, pixbeam.b * FWHM2CC), max(sy, sx) * 1.25]

            # Set up the parameters for the fit, including constraints
            prefix = "c{0}_".format(i)
            params.add(prefix + 'amp', value=src.peak_flux, vary=True)
            # for now the xo/yo are locations within the main image, we correct this later
            params.add(prefix + 'xo', value=source_x, min=source_x-sx/2., max=source_x+sx/2., vary=stage>=2)
            params.add(prefix + 'yo', value=source_y, min=source_y-sy/2., max=source_y+sy/2., vary=stage>=2)
            params.add(prefix + 'sx', value=sx, min=s_lims[0], max=s_lims[1], vary=stage>=3)
            params.add(prefix + 'sy', value=sy, min=s_lims[0], max=s_lims[1], vary=stage>=3)
            params.add(prefix + 'theta', value=theta, vary=stage>=3)
            params.add(prefix + 'flags', value=0, vary=False)
            # this source is being refit so add it to the list
            included_sources.append(src)
            i += 1

            # TODO: Allow this mask to be used in conjunction with the FWHM mask that is defined further on
            # # Use pixels above outerclip sigmas..
            # if outerclip>=0:
            #     mask = np.where(data[xmin:xmax,ymin:ymax]-outerclip*rmsimg[xmin:xmax,ymin:ymax]>0)
            # else: # negative outer clip means use all the pixels
            #     mask = np.where(data[xmin:xmax,ymin:ymax])
            #
            # # convert the pixel indices to be pixels within the parent data set
            # xmask = mask[0] + xmin
            # ymask = mask[1] + ymin
            # island_mask.extend(zip(xmask,ymask))

        if i==0:
            log.debug("No sources found in island {0}".format(src.island))
            continue
        params.add('components', value=i, vary=False)
        # params.components = i
        log.debug(" {0} components being fit".format(i))
        # now we correct the xo/yo positions to be relative to the sub-image
        log.debug("xmxxymyx {0} {1} {2} {3}".format(xmin, xmax, ymin, ymax))
        for i in range(params['components'].value):
            try:
                prefix = "c{0}_".format(i)
                params[prefix + 'xo'].value -= xmin
                params[prefix + 'xo'].min -= xmin
                params[prefix + 'xo'].max -= xmin
                params[prefix + 'yo'].value -= ymin
                params[prefix + 'yo'].min -= ymin
                params[prefix + 'yo'].max -= ymin
            except Exception, e:
                log.error(" ARG !")
                log.info(params)
                log.info(params['components'].value)
                log.info("trying to access component {0}".format(i))
                raise e
        # log.debug(params)
        # don't fit if there are no sources
        if params['components'].value<1:
            log.info("Island {0} has no components".format(src.island))
            continue

        # this .copy() will stop us from modifying the parent region when we later apply our mask.
        idata = data[xmin:xmax, ymin:ymax].copy()
        # now convert these back to indices within the idata region
        # island_mask = np.array([(x-xmin, y-ymin) for x, y in island_mask])

        allx, ally = np.indices(idata.shape)
        # mask to include pixels that are withn the FWHM of the sources being fit
        mask_params = copy.deepcopy(params)
        for i in range(mask_params['components'].value):
            prefix = 'c{0}_'.format(i)
            mask_params[prefix+'amp'].value = 1
        mask_model = ntwodgaussian_lmfit(mask_params)
        mask = np.where(mask_model(allx.ravel(), ally.ravel()) <= 0.1)
        mask = allx.ravel()[mask], ally.ravel()[mask]
        del mask_params

        idata[mask] = np.nan

        mx, my = np.where(np.isfinite(idata))
        non_nan_pix = len(mx)

        log.debug("island extracted:")
        log.debug(" x[{0}:{1}] y[{2}:{3}]".format(xmin, xmax, ymin, ymax))
        log.debug(" max = {0}".format(np.nanmax(idata)))
        log.debug(" total {0}, masked {1}, not masked {2}".format(len(allx), len(allx)-non_nan_pix, non_nan_pix))

        # Check to see that each component has some data within the central 3x3 pixels of it's location
        # If not then we don't fit that component
        for i in range(params['components'].value):
            prefix = "c{0}_".format(i)
            # figure out a box around the center of this
            cx, cy = params[prefix+'xo'].value, params[prefix+'yo'].value  # central pixel coords
            log.debug(" comp {0}".format(i))
            log.debug("  cx,cy {0} {1}".format(cx ,cy))
            xmx, xmn = np.clip(cx+2, 0, idata.shape[0]), np.clip(cx-1, 0, idata.shape[0])
            ymx, ymn = np.clip(cy+2, 0, idata.shape[1]), np.clip(cy-1, 0, idata.shape[1])
            square = idata[xmn:xmx, ymn:ymx]
            # if there are no not-nan pixels in this region then don't vary any parameters
            if not np.any(np.isfinite(square)):
                log.debug(" not fitting component {0}".format(i))
                params[prefix+'amp'].value = np.nan
                for p in ['amp', 'xo', 'yo', 'sx', 'sy', 'theta']:
                    params[prefix+p].vary = False
                    params[prefix+p].stderr = np.nan  # this results in an error of -1 later on
                params[prefix+'flags'].value |= flags.NOTFIT

        # determine the number of free parameters and if we have enough data for a fit
        nfree = np.count_nonzero([params[p].vary for p in params.keys()])
        log.debug(params)
        if nfree < 1:
            log.debug(" Island has no components to fit")
            result = DummyLM()
            model = params
        else:
            if non_nan_pix < nfree:
                log.debug("More free parameters {0} than available pixels {1}".format(nfree, non_nan_pix))
                if non_nan_pix >= params['components'].value:
                    log.debug("Fixing all parameters except amplitudes")
                    for p in params.keys():
                        if 'amp' not in p:
                            params[p].vary = False
                else:
                    log.debug(" no not-masked pixels, skipping".format(src.island, src.source))
                continue

            # do the fit
            # if the pixel beam is not valid, then recalculate using the location of the last source to have a valid psf
            if pixbeam is None:
                if src_valid_psf is not None:
                    pixbeam = global_data.psfhelper.get_pixbeam(src_valid_psf.ra,src_valid_psf.dec)
                else:
                    log.critical("Cannot determine pixel beam")
            fac = 1/np.sqrt(2)
            C = Cmatrix(mx, my, pixbeam.a*FWHM2CC*fac, pixbeam.b*FWHM2CC*fac, pixbeam.pa)
            B = Bmatrix(C)
            errs = np.nanmax(rmsimg[xmin:xmax, ymin:ymax])
            result, _ = do_lmfit(idata, params, B=B)
            model = covar_errors(result.params, idata, errs=errs, B=B, C=C)


        # convert the results to a source object
        offsets = (xmin, xmax, ymin, ymax)
        # TODO allow for island fluxes in the refitting.
        island_data = IslandFittingData(inum, i=idata, offsets=offsets, doislandflux=False, scalars=(4,4,None))
        new_src = result_to_components(result, model, island_data, src.flags)

        # preserve the uuid so we can do exact matching between catalogs
        for ns, s in zip(new_src, included_sources):
            ns.uuid = s.uuid
            # if the position wasn't fit then copy the errors from the input catalog
            if stage<2:
                ns.err_ra = s.err_ra
                ns.err_dec = s.err_dec
            # if the shape wasn't fit then copy the errors from the input catalog
            if stage <3:
                ns.err_a = s.err_a
                ns.err_b = s.err_b
                ns.err_pa = s.err_pa
        sources.extend(new_src)
    return sources


def fit_island(island_data):
    """
    Take an Island, do all the parameter estimation and fitting.

    :param island_data: an IslandFittingData object
    :return: a list of sources that are within the island
    """
    global global_data

    # global data
    dcurve = global_data.dcurve
    rmsimg = global_data.rmsimg
    beam = global_data.beam

    # island data
    isle_num = island_data.isle_num
    idata = island_data.i
    innerclip, outerclip, max_summits = island_data.scalars
    xmin, xmax, ymin, ymax = island_data.offsets

    icurve = dcurve[xmin:xmax, ymin:ymax]
    rms = rmsimg[xmin:xmax, ymin:ymax]

    is_flag = 0
    pixbeam = global_data.psfhelper.get_pixbeam_pixel((xmin+xmax)/2., (ymin+ymax)/2.)
    if pixbeam is None:
        # This island is not 'on' the sky, ignore it
        return []

    log.debug("=====")
    log.debug("Island ({0})".format(isle_num))

    params = estimate_lmfit_parinfo(idata, rms, icurve, beam, innerclip, outerclip, offsets=[xmin, ymin],
                               max_summits=max_summits)

    # islands at the edge of a region of nans
    # result in no components
    if params is None or params['components'].value <1:
        return []

    log.debug("Rms is {0}".format(np.shape(rms)))
    log.debug("Isle is {0}".format(np.shape(idata)))
    log.debug(" of which {0} are masked".format(sum(np.isnan(idata).ravel() * 1)))

    # Check that there is enough data to do the fit
    mx, my = np.where(np.isfinite(idata))
    non_blank_pix = len(mx)
    free_vars = len( [ 1 for a in params.keys() if params[a].vary])
    if non_blank_pix < free_vars or free_vars ==0:
        log.debug("Island {0} doesn't have enough pixels to fit the given model".format(isle_num))
        log.debug("non_blank_pix {0}, free_vars {1}".format(non_blank_pix,free_vars))
        result = DummyLM()
        model = params
        is_flag |= flags.NOTFIT
    else:
        # Model is the fitted parameters
        fac = 1/np.sqrt(2)
        C = Cmatrix(mx, my, pixbeam.a*FWHM2CC*fac, pixbeam.b*FWHM2CC*fac, pixbeam.pa)
        B = Bmatrix(C)
        ## For testing the fitting without the inverse co-variance matrix, set these to None
        # B = None
        # C = None
        log.debug("C({0},{1},{2},{3},{4})".format(len(mx),len(my),pixbeam.a*FWHM2CC, pixbeam.b*FWHM2CC, pixbeam.pa))
        errs = np.nanmax(rms)
        result, _ = do_lmfit(idata, params, B=B)
        if not result.errorbars:
            is_flag |= flags.FITERR
        # get the real (sky) parameter errors
        model = covar_errors(result.params, idata, errs=errs, B=B, C=C)

        if not result.success:
            is_flag |= flags.FITERR

    log.debug(model)

    # convert the fitting results to a list of sources [and islands]
    sources =result_to_components(result, model, island_data, is_flag)

    return sources


def fit_islands(islands):
    """
    Execute fitting on a list of islands
    This function just wraps around fit_island, so that when we do multiprocesing
    a single process will fit multiple islands before returning results.

    :param islands: a list of IslandFittingData objects
    :return: a list of OutputSources
    """
    log.debug("Fitting group of {0} islands".format(len(islands)))
    sources = []
    for island in islands:
        res = fit_island(island)
        sources.extend(res)
    return sources


def find_sources_in_image(filename, hdu_index=0, outfile=None, rms=None, max_summits=None, csigma=None, innerclip=5,
                          outerclip=4, cores=None, rmsin=None, bkgin=None, beam=None, doislandflux=False,
                          returnrms=False, nopositive=False, nonegative=False, mask=None, lat=None, imgpsf=None):
    """
    Run the Aegean source finder.

    :param filename: image filename or HDUList
    :param hdu_index: the index of the FITS HDU (extension)
    :param outfile: file for printing catalog (NOT a table, just a text file of my own design)
    :param rms: use this rms for the entire image (will also assume that background is 0)
    :param max_summits: fit up to this many components to each island (extras are included but not fit)
    :param csigma: use this as the clipping limit for the curvature map (not really used/tested)
    :param innerclip: the seeding clip, in sigmas, for seeding islands of pixels
    :param outerclip: the flood clip in sigmas, used for flooding islands
    :param cores: number of CPU cores to use. None means all cores.
    :param rmsin: filename or HDUList for an rms/noise image.
                  If None, aegean will use the Zones algorithm to calculate one
    :param bkgin: filename or HDUList for a background image
                  If None, aegean will use the Zones algorithm to calculate one
    :param beam: (major,minor,pa) (all degrees) of the synthesised beam to use.
                  Replaces whatever is given in the FITS header.
                  If the FITS header has no BMAJ/BMIN then this is required.
    :param doislandflux: calculate the properties of the islands as well as the components
    :param returnrms: if true, return the rms image
    :param nopositive: if true, sources with positive fluxes will not be reported
    :param nonegative: if true, sources with negative fluxes will not be reported
    :param mask: the filename of a region file created by MIMAS.
                 Islands outside of this region will be ignored.
    :param lat: The latitude of the telescope (or declination of zenith)
    :param imgpsf: filename or HDUList for a psf image.
    :return: list of sources [and an rms image if returnrms is True]
    """
    global log
    if log is None:
        log = logging.getLogger()

    # Tell numpy to be quiet
    np.seterr(invalid='ignore')
    if cores is not None:
        assert (cores >= 1), "cores must be one or more"

    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, beam=beam, rms=rms, cores=cores,
                 csigma=csigma, verb=True, mask=mask, lat=lat, psf=imgpsf)

    rmsimg = global_data.rmsimg
    data = global_data.data_pix
    beam = global_data.beam

    log.info("beam = {0:5.2f}'' x {1:5.2f}'' at {2:5.2f}deg".format(beam.a * 3600, beam.b * 3600, beam.pa))
    log.info("seedclip={0}".format(innerclip))
    log.info("floodclip={0}".format(outerclip))

    isle_num = 0

    if cores == 1:  #single-threaded, no parallel processing
        queue = []
    else:
        # This check is also made during start up when running aegean from the command line
        # However I reproduce it here so that we don't fall over when aegean is being imported
        # into other codes (eg the VAST pipeline)
        if cores is None:
            cores = multiprocessing.cpu_count()
            log.info("Found {0} cores".format(cores))
        log.info("Using {0} subprocesses".format(cores))
        try:
            queue = pprocess.Queue(limit=cores, reuse=1)
            fit_parallel = queue.manage(pprocess.MakeReusable(fit_islands))
        except AttributeError, e:
            if 'poll' in e.message:
                log.warn("Your O/S doesn't support select.poll(): Reverting to cores=1")
                cores = 1
                queue = []
            else:
                raise e

    sources = []

    if outfile:
        print >> outfile, header.format("{0}-({1})".format(__version__,__date__), filename)
        print >> outfile, OutputSource.header
    island_group = []
    group_size = 20
    for i, xmin, xmax, ymin, ymax in gen_flood_wrap(data, rmsimg, innerclip, outerclip, domask=True):
        if len(i) <= 1:
            # empty islands have length 1
            continue
        isle_num += 1
        scalars = (innerclip, outerclip, max_summits)
        offsets = (xmin, xmax, ymin, ymax)
        island_data = IslandFittingData(isle_num, i, scalars, offsets, doislandflux)
        # If cores==1 run fitting in main process. Otherwise build up groups of islands
        # and submit to queue for subprocesses. Passing a group of islands is more
        # efficient than passing single islands to the subprocesses.
        if cores == 1:
            res = fit_island(island_data)
            queue.append(res)
        else:
            island_group.append(island_data)
            # If the island group is full queue it for the subprocesses to fit
            if len(island_group) >= group_size:
                fit_parallel(island_group)
                island_group = []

    # The last partially-filled island group also needs to be queued for fitting
    if len(island_group) > 0:
        fit_parallel(island_group)

    for srcs in queue:
        if srcs:  # ignore empty lists
            for src in srcs:
                # ignore sources that we have been told to ignore
                if (src.peak_flux > 0 and nopositive) or (src.peak_flux < 0 and nonegative):
                    continue
                sources.append(src)
    if outfile:
        components, islands, simples = classify_catalog(sources)
        for source in sorted(components):
            outfile.write(str(source))
            outfile.write("\n")
    if returnrms:
        return [sources, global_data.rmsimg]
    else:
        return sources


def priorized_fit_islands(filename, catfile, hdu_index=0, outfile=None, bkgin=None, rmsin=None, cores=1, rms=None,
                           beam=None, lat=None, imgpsf=None, catpsf=None, stage=3, ratio=1.0, outerclip=3, doregroup=True):
    """
    Take an input catalog, and image, and optional background/noise images
    fit the flux and ra/dec for each of the given sources, keeping the morphology fixed

    if doregroup is true the groups will be recreated based on a matching radius/probability.
    if doregroup is false then the islands of the input catalog will be preserved.

    Multiple cores can be specified, and will be used.

    :param filename: image file or hdu
    :param catfile: catalog file name
    :param hdu_index:
    :param outfile: output file for ascii output (not tables)
    :param bkgin: background image name
    :param rmsin: noise image name
    :param cores: number of cores to use
    :param rms: if not none, then use this constant rms for the entire image
    :param beam: beam parameters to be used instead of those that may be in the fits header
    :param lat: latitude of telescope
    :param imgpsf: a psf map that corresponds to the input image
    :param catpsf: a psf map that corresponds to the input catalog
    :param stage: refitting stage
    :param ratio: ratio of image psf to catalog psf
    :param outerclip: pixels above an snr of this amount will be used in fitting, <0 -> all pixels.
    :param doregroup:  True - doregroup, False - use island data for groups
    :return: a list of source objects
    """

    from AegeanTools.cluster import regroup
    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, verb=True,
                 do_curve=False, beam=beam, lat=lat, psf=imgpsf)

    beam = global_data.beam
    far = 10*beam.a  # degrees
    # load the table and convert to an input source list
    input_table = load_table(catfile)
    input_sources = np.array(table_to_source_list(input_table))
    src_mask = np.ones(len(input_sources), dtype=bool)

    # the input sources are the initial conditions for our fits.
    # Expand each source size if needed.
    if catpsf is not None:
        log.info("Using catalog PSF from {0}".format(catpsf))
        psf_helper = PSFHelper(catpsf, None)  # might need to set the WCSHelper to be not None
        for i, src in enumerate(input_sources):
            catbeam = psf_helper.get_beam(src.ra, src.dec)
            imbeam = global_data.psfhelper.get_beam(src.ra, src.dec)
            # If either of the above are None then we skip this source.
            if catbeam is None or imbeam is None:
                src_mask[i] = False
                log.info("Excluding source ({0.island},{0.source}) due to lack of psf knowledge".format(src))
                continue
            src.a = (src.a/3600)**2 - catbeam.a**2 + imbeam.a**2  # degrees
            if src.a < 0:
                src.a = imbeam.a*3600  # arcsec
            else:
                src.a = np.sqrt(src.a)*3600  # arcsec

            src.b = (src.b/3600)**2 - catbeam.b**2 + imbeam.b**2
            if src.b < 0:
                src.b = imbeam.b*3600  # arcsec
            else:
                src.b = np.sqrt(src.b)*3600  # arcsec

    elif ratio is not None:
        log.info("Using ratio of {0} to scale input source shapes".format(ratio))
        far *= ratio
        for i, src in enumerate(input_sources):
            skybeam = global_data.psfhelper.get_beam(src.ra, src.dec)
            if skybeam is None:
                src_mask[i] = False
                continue
            src.a = np.sqrt(src.a**2 + (skybeam.a*3600)**2*(1-1/ratio**2))
            src.b = np.sqrt(src.b**2 + (skybeam.b*3600)**2*(1-1/ratio**2))
            # source with funky a/b are also rejected
            if not np.all(np.isfinite((src.a,src.b))):
                src_mask[i] = False
    else:
        log.info("Not scaling input source sizes")

    log.info("{0} sources in catalog".format(len(input_sources)))
    log.info("{0} sources accepted".format(sum(src_mask)))
    input_sources = input_sources[src_mask]
    # redo the grouping if required
    if doregroup:
        groups = regroup(input_sources, eps=np.sqrt(2), far=far)
    else:
        groups = list(island_itergen(input_sources))

    if cores == 1:  # single-threaded, no parallel processing
        queue = []
    else:
        # This check is also made during start up when running aegean from the command line
        # However I reproduce it here so that we don't fall over when aegean is being imported
        # into other codes (eg the VAST pipeline)
        if cores is None:
            cores = multiprocessing.cpu_count()
            log.info("Found {0} cores".format(cores))
        else:
            log.info("Using {0} subprocesses".format(cores))
        try:
            queue = pprocess.Queue(limit=cores, reuse=1)
            fit_parallel = queue.manage(pprocess.MakeReusable(refit_islands))
        except AttributeError, e:
            if 'poll' in e.message:
                log.warn("Your O/S doesn't support select.poll(): Reverting to cores=1")
                cores = 1
                queue = []
            else:
                raise e

    sources = []

    island_group = []
    group_size = 20

    for i, island in enumerate(groups):
        island_group.append(island)
        # If the island group is full queue it for the subprocesses to fit
        if len(island_group) >= group_size:
            if cores > 1:
                fit_parallel(island_group, stage, outerclip, istart=i)
            else:
                res = refit_islands(island_group, stage, outerclip, istart=i)
                queue.append(res)
            island_group = []

    # The last partially-filled island group also needs to be queued for fitting
    if len(island_group) > 0:
        if cores > 1:
            fit_parallel(island_group, stage, outerclip, istart=i)
        else:
            res = refit_islands(island_group, stage, outerclip, istart=i)
            queue.append(res)

    # now unpack the fitting results in to a list of sources
    for s in queue:
        sources.extend(s)

    sources = sorted(sources)

    # Write the output to the output file (note that None -> stdout)
    if outfile:
        print >> outfile, header.format("{0}-({1})".format(__version__,__date__), filename)
        print >> outfile, OutputSource.header

    components = 0
    for source in sources:
        if type(source) == OutputSource:
            components +=1
            print >> outfile, str(source)

    log.info("fit {0} components".format(components))
    return sources


def island_itergen(catalog):
    """
    Iterate over a catalog of sources, and return an island worth of sources at a time.
    Yields a list of components, one island at a time

    :param catalog: A list of objects which have island/source attributes
    :return:
    """
    # reverse sort so that we can pop the last elements and get an increasing island number
    catalog = sorted(catalog)
    catalog.reverse()
    group = []

    # using pop and keeping track of the list length ourselves is faster than
    # constantly asking for len(catalog)
    src = catalog.pop()
    c_len = len(catalog)
    isle_num = src.island
    while c_len >= 0:
        if src.island == isle_num:
            group.append(src)
            c_len -= 1
            if c_len <0:
                # we have just added the last item from the catalog
                # and there are no more to pop
                yield group
            else:
                src = catalog.pop()
        else:
            isle_num += 1
            # maybe there are no sources in this island so skip it
            if group == []:
                continue
            yield group
            group = []
    return


def force_measure_flux(radec):
    """
    Measure the flux of a point source at each of the specified locations
    Not fitting is done, just forced measurements
    Assumes that global_data has been populated

    :param radec: the locations at which to measure fluxes
    :return: [(flux,err),...] corresponding to each ra/dec
    """
    from AegeanTools.fitting import ntwodgaussian_mpfit
    catalog = []

    dummy = SimpleSource()
    dummy.peak_flux = np.nan
    dummy.peak_pixel = np.nan
    dummy.flags = flags.FITERR

    shape = global_data.data_pix.shape

    if global_data.wcshelper.lat is not None:
        log.warn("No account is being made for telescope latitude, even though it has been supplied")
    for ra, dec in radec:
        # find the right pixels from the ra/dec
        source_x, source_y = global_data.wcshelper.sky2pix([ra, dec])
        x = int(round(source_x))
        y = int(round(source_y))

        # reject sources that are outside the image bounds, or which have nan data/rms values
        if not 0 <= x < shape[0] or not 0 <= y < shape[1] or \
                not np.isfinite(global_data.data_pix[x, y]) or \
                not np.isfinite(global_data.rmsimg[x, y]):
            catalog.append(dummy)
            continue

        flag = 0
        # make a pixbeam at this location
        pixbeam = global_data.psfhelper.get_pixbeam(ra,dec)
        if pixbeam is None:
            flag |= flags.WCSERR
            pixbeam = Beam(1, 1, 0)
        # determine the x and y extent of the beam
        xwidth = 2 * pixbeam.a * pixbeam.b
        xwidth /= np.hypot(pixbeam.b * np.sin(np.radians(pixbeam.pa)), pixbeam.a * np.cos(np.radians(pixbeam.pa)))
        ywidth = 2 * pixbeam.a * pixbeam.b
        ywidth /= np.hypot(pixbeam.b * np.cos(np.radians(pixbeam.pa)), pixbeam.a * np.sin(np.radians(pixbeam.pa)))
        # round to an int and add 1
        ywidth = int(round(ywidth)) + 1
        xwidth = int(round(xwidth)) + 1

        # cut out an image of this size
        xmin = max(0, x - xwidth / 2)
        ymin = max(0, y - ywidth / 2)
        xmax = min(shape[0], x + xwidth / 2 + 1)
        ymax = min(shape[1], y + ywidth / 2 + 1)
        data = global_data.data_pix[xmin:xmax, ymin:ymax]

        # Make a Gaussian equal to the beam with amplitude 1.0 at the position of the source
        # in terms of the pixel region.
        amp = 1.0
        xo = source_x - xmin
        yo = source_y - ymin
        params = [amp, xo, yo, pixbeam.a * FWHM2CC, pixbeam.b * FWHM2CC, pixbeam.pa]
        gaussian_data = ntwodgaussian_mpfit(params)(*np.indices(data.shape))

        # Calculate the "best fit" amplitude as the average of the implied amplitude
        # for each pixel. Error is stddev.
        # Only use pixels within the FWHM, ie value>=0.5. Set the others to NaN
        ratios = np.where(gaussian_data >= 0.5, data / gaussian_data, np.nan)
        flux, error = gmean(ratios)

        # sources with fluxes or flux errors that are not finite are not valid
        # an error of identically zero is also not valid.
        if not np.isfinite(flux) or not np.isfinite(error) or error == 0.0:
            catalog.append(dummy)
            continue

        source = SimpleSource()
        source.ra = ra
        source.dec = dec
        source.peak_flux = flux
        source.err_peak_flux = error
        source.background = global_data.bkgimg[x, y]
        source.flags = flag
        source.peak_pixel = np.nanmax(data)
        source.local_rms = global_data.rmsimg[x, y]
        source.a = global_data.beam.a
        source.b = global_data.beam.b
        source.pa = global_data.beam.pa

        catalog.append(source)
        if logging.getLogger('Aegean').isEnabledFor(logging.DEBUG):
            log.debug("Measured source {0}".format(source))
            log.debug("  used area = [{0}:{1},{2}:{3}]".format(xmin, xmax, ymin, ymax))
            log.debug("  xo,yo = {0},{1}".format(xo, yo))
            log.debug("  params = {0}".format(params))
            log.debug("  flux at [xmin+xo,ymin+yo] = {0} Jy".format(data[int(xo), int(yo)]))
            log.debug("  error = {0}".format(error))
            log.debug("  rms = {0}".format(source.local_rms))
    return catalog


def measure_catalog_fluxes(filename, catfile, hdu_index=0, outfile=None, bkgin=None, rmsin=None, cores=1, rms=None,
                           beam=None, lat=None):
    """
    Measure the flux at a given set of locations, assuming point sources.

    This function is of limited use, priorized_fit_islands should be used instead.

    :param filename: filename or HDUList of image
    :param catfile: a catalog of source positions (ra,dec)
    :param hdu_index: if fits file has more than one hdu, it can be specified here
    :param outfile: the output file to write to (NOT a table)
    :param bkgin: a background image filename or HDUList
    :param rmsin: an rms image filename or HDUList
    :param cores: cores to use
    :param rms: forced rms value
    :param beam: beam parameters to override those given in fits header
    :param lat: telescope latitude (ignored)
    :return: a list of simple sources
    """

    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, verb=True,
                 do_curve=False, beam=beam, lat=lat)

    # load catalog
    radec = load_catalog(catfile)
    # measure fluxes
    sources = force_measure_flux(radec)
    # write output
    print >> outfile, header.format("{0}-({1})".format(__version__,__date__), filename)
    print >> outfile, SimpleSource.header
    for source in sources:
        print >> outfile, str(source)
    return sources


def VASTP_measure_catalog_fluxes(filename, positions, hdu_index=0, bkgin=None, rmsin=None,
                                 rms=None, cores=1, beam=None, debug=False):
    """
    A version of measure_catalog_fluxes that will accept a list of positions instead of reading from a file.
    Input:
        filename - fits image file name to be read
        positions - a list of source positions (ra,dec)
        hdu_index - if fits file has more than one hdu, it can be specified here
        outfile - the output file to write to
        bkgin - a background image filename
        rmsin - an rms image filename
        cores - cores to use
        rms - forced rms value
        beam - beam parameters to override those given in fits header
    """
    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, beam=beam, verb=True,
                 do_curve=False)
    # measure fluxes
    if debug:
        level = log.getLogger().getEffectiveLevel()
        log.getLogger().setLevel(log.DEBUG)
    sources = force_measure_flux(positions)
    if debug:
        log.getLogger().setLevel(level)
    return sources


def VASTP_refit_sources(filename, sources, hdu_index=0, bkgin=None, rmsin=None, rms=None, cores=1, beam=None, debug=False):
    """
    A version of priorized_fit_islands that will work with the vast pipeline
    Input:
        filename - fits image file name to be read
        sources - a list of source objects
        hdu_index - if fits file has more than one hdu, it can be specified here
        outfile - the output file to write to
        bkgin - a background image filename
        rmsin - an rms image filename
        cores - cores to use
        rms - forced rms value
        beam - beam parameters to override those given in fits header
    """
    logging.info(" refitting {0} sources".format(len(sources)))
    if len(sources)<1:
        return []
    stage = 1
    outerclip = 4 # ultimately ignored but required for now

    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, verb=True,
                 do_curve=False, beam=beam)

    new_sources = []

    for src in sources:
        res = refit_islands([[src]], stage, outerclip)
        # if the source is not able to be fit then we dummy the source
        if len(res)<1:
            d = OutputSource()
            d.peak_flux = np.nan
            res = [d]
        elif len(res)!=2:
            logging.error("expecting two sources, but got {0}".format(len(res)))
        s = res[0]
        s.flags = 0
        # convert a/b from deg->arcsec to emulate a mistake that was made in forcemeasurements
        # and then corrected in the vast pipeline (ugh).
        s.a /=3600
        s.b /=3600

        new_sources.append(s)

    logging.info("Returning {0} sources".format(len(new_sources)))
    return new_sources


def save_background_files(image_filename, hdu_index=0, bkgin=None, rmsin=None, beam=None, rms=None, cores=1,
                          outbase=None):
    """
    Generate and save the background and RMS maps as FITS files.
    They are saved in the current directly as aegean-background.fits and aegean-rms.fits.

    :param image_filename: filename or HDUList of image
    :param hdu_index: if fits file has more than one hdu, it can be specified here
    :param bkgin: a background image filename or HDUList
    :param rmsin: an rms image filename or HDUList
    :param beam: beam parameters to override those given in fits header
    :param rms: forced rms value
    :param cores: cores to use
    :param outbase: basename for output files
    :return:
    """
    global global_data

    log.info("Saving background / RMS maps")
    # load image, and load/create background/rms images
    load_globals(image_filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, beam=beam, verb=True, rms=rms,
                 cores=cores, do_curve=True)
    img = global_data.img
    bkgimg, rmsimg = global_data.bkgimg, global_data.rmsimg
    curve = np.array(global_data.dcurve,dtype=np.float32)
    # mask these arrays have the same mask the same as the data
    mask = np.where(np.isnan(global_data.data_pix))
    bkgimg[mask] = np.NaN
    rmsimg[mask] = np.NaN
    curve[mask] = np.NaN

    # Generate the new FITS files by copying the existing HDU and assigning new data.
    # This gives the new files the same WCS projection and other header fields.
    new_hdu = img.hdu
    # Set the ORIGIN to indicate Aegean made this file
    new_hdu.header["ORIGIN"] = "Aegean {0}-({1})".format(__version__,__date__)
    for c in ['CRPIX3', 'CRPIX4', 'CDELT3', 'CDELT4', 'CRVAL3', 'CRVAL4', 'CTYPE3', 'CTYPE4']:
        if c in new_hdu.header:
            del new_hdu.header[c]

    if outbase is None:
        outbase, _ = os.path.splitext(os.path.basename(image_filename))
    noise_out = outbase + '_rms.fits'
    background_out = outbase + '_bkg.fits'
    curve_out = outbase +'_crv.fits'

    new_hdu.data = bkgimg
    new_hdu.writeto(background_out, clobber=True)
    log.info("Wrote {0}".format(background_out))

    new_hdu.data = rmsimg
    new_hdu.writeto(noise_out, clobber=True)
    log.info("Wrote {0}".format(noise_out))

    new_hdu.data = curve
    new_hdu.writeto(curve_out, clobber = True)
    log.info("Wrote {0}".format(curve_out))
    return


if __name__ == "__main__":
    usage = "usage: %prog [options] FileName.fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--find", dest='find', action='store_true', default=False,
                      help='Source finding mode. [default: true, unless --save or --measure are selected]')
    parser.add_option("--cores", dest="cores", type="int", default=None,
                      help="Number of CPU cores to use for processing [default: all cores]")
    parser.add_option("--debug", dest="debug", action="store_true", default=False,
                      help="Enable debug mode. [default: false]")
    parser.add_option("--hdu", dest="hdu_index", type="int", default=0,
                      help="HDU index (0-based) for cubes with multiple images in extensions. [default: 0]")
    parser.add_option("--out", dest='outfile', default=None,
                      help="Destination of Aegean catalog output. [default: stdout]")
    parser.add_option("--table", dest='tables', default=None,
                      help="Additional table outputs, format inferred from extension. [default: none]")
    parser.add_option("--tformats",dest='table_formats', action="store_true",default=False,
                      help='Show a list of table formats supported in this install, and their extensions')
    parser.add_option("--forcerms", dest='rms', type='float', default=None,
                      help="Assume a single image noise of rms, and a background of zero. [default: false]")
    parser.add_option("--noise", dest='noiseimg', default=None,
                      help="A .fits file that represents the image noise (rms), created from Aegean with --save " +
                           "or BANE. [default: none]")
    parser.add_option('--background', dest='backgroundimg', default=None,
                      help="A .fits file that represents the background level, created from Aegean with --save " +
                           "or BANE. [default: none]")
    parser.add_option('--psf', dest='imgpsf',default=None,
                      help="A .fits file that represents the size (degrees) of a blurring disk. " +
                           "This disk is convolved with the BMAJ/BMIN listed in the FITS header and " +
                           "the result becomes the local PSF.")

    parser.add_option('--autoload', dest='autoload', action="store_true", default=False,
                      help="Automatically look for background, noise, region, and psf files using the input filename as a hint. [default: don't do this]")
    parser.add_option("--maxsummits", dest='max_summits', type='float', default=None,
                      help="If more than *maxsummits* summits are detected in an island, no fitting is done, only estimation. [default: no limit]")
    parser.add_option('--seedclip', dest='innerclip', type='float', default=5,
                      help='The clipping value (in sigmas) for seeding islands. [default: 5]')
    parser.add_option('--floodclip', dest='outerclip', type='float', default=4,
                      help='The clipping value (in sigmas) for growing islands. [default: 4]')
    parser.add_option('--beam', dest='beam', type='float', nargs=3, default=None,
                      help='The beam parameters to be used is "--beam major minor pa" all in degrees. [default: read from fits header].')
    parser.add_option('--telescope', dest='telescope', type=str, default=None,
                      help='The name of the telescope used to collect data. [MWA|VLA|ATCA|LOFAR]')
    parser.add_option('--lat', dest='lat', type=float, default=None,
                      help='The latitude of the telescope used to collect data.')
    parser.add_option('--versions', dest='file_versions', action="store_true", default=False,
                      help='Show the file versions of relevant modules. [default: false]')
    parser.add_option('--island', dest='doislandflux', action="store_true", default=False,
                      help='Also calculate the island flux in addition to the individual components. [default: false]')
    parser.add_option('--nopositive', dest='nopositive', action="store_true", default=False,
                      help="Don't report sources with positive fluxes. [default: false]")
    parser.add_option('--negative', dest='negative', action="store_true", default=False,
                      help="Report sources with negative fluxes. [default: false]")

    parser.add_option('--region', dest='region', default=None,
                      help="Use this regions file to restrict source finding in this image.")

    parser.add_option('--save', dest='save', action="store_true", default=False,
                      help='Enable the saving of the background and noise images. Sets --find to false. [default: false]')
    parser.add_option('--outbase', dest='outbase', default=None,
                      help='If --save is True, then this specifies the base name of the background and noise images. [default: inferred from input image]')

    parser.add_option('--measure', dest='measure', action='store_true', default=False,
                      help='Enable forced measurement mode. Requires an input source list via --input. Sets --find to false. [default: false]')
    parser.add_option('--priorized', dest='priorized', default=0, type=int,
                      help="Enable priorized fitting, with stage = n [default=1]")
    parser.add_option('--ratio', dest='ratio', default=None, type=float,
                      help="The ratio of synthesized beam sizes (image psf / input catalog psf). For use with priorized.")
    parser.add_option('--noregroup',dest='regroup', default=True, action='store_false',
                      help='Do not regroup islands before priorized fitting.')
    parser.add_option('--input', dest='input', default=None,
                      help='If --measure is true, this gives the filename for a catalog of locations at which fluxes will be measured. [default: none]')
    parser.add_option('--catpsf', dest='catpsf', default=None,
                      help='A psf map corresponding to the input catalog. This will allow for the correct resizing of '+
                           'sources when the catalog and image psfs differ.')

    (options, args) = parser.parse_args()

    # configure logging
    logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
    log = logging.getLogger("Aegean")
    logging_level = logging.DEBUG if options.debug else logging.INFO
    log.setLevel(logging_level)
    log.info("This is Aegean {0}-({1})".format(__version__,__date__))

    if options.table_formats:
        show_formats()
        sys.exit(0)

    if options.file_versions:
        log.info("Numpy {0} from {1} ".format(np.__version__, np.__file__))
        log.info("Scipy {0} from {1}".format(scipy.__version__, scipy.__file__))
        log.info("AstroPy {0} from {1}".format(astropy.__version__, astropy.__file__))
        log.info("LMFit {0} from {1}".format(lmfit.__version__, lmfit.__file__))
        try:
            import h5py
            log.info("h5py {0} from {1}".format(h5py.__version__, h5py.__file__))
        except ImportError:
            log.info("h5py not found")
        sys.exit(0)

    # print help if the user enters no options or filename
    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    # check that a valid filename was entered
    filename = args[0]
    if not os.path.exists(filename):
        log.error("{0} not found".format(filename))
        sys.exit(1)

    # tell numpy to shut up about "invalid values encountered"
    # Its just NaN's and I don't need to hear about it once per core
    np.seterr(invalid='ignore', divide='ignore')

    # check for nopositive/negative conflict
    if options.nopositive and not options.negative:
        log.warning('Requested no positive sources, but no negative sources. Nothing to find.')
        sys.exit()

    # if measure/save are enabled we turn off "find" unless it was specifically set
    if (options.measure or options.save or options.priorized) and not options.find:
        options.find = False
    else:
        options.find = True

    # debugging in multi core mode is very hard to understand
    if options.debug:
        log.info("Setting cores=1 for debugging")
        options.cores = 1

    # check/set cores to use
    if options.cores is None:
        options.cores = multiprocessing.cpu_count()
        log.info("Found {0} cores".format(options.cores))
    if options.cores > 1:
        try:
            queue = pprocess.Queue(limit=options.cores, reuse=1)
            temp = queue.manage(pprocess.MakeReusable(fit_islands))
        except AttributeError, e:
            if 'poll' in e.message:
                log.warn("Your O/S doesn't support select.poll(): Reverting to cores=1")
                options.cores = 1
                queue = None
                temp = None
            else:
                log.error("Your system can't seem to make a queue, try using --cores=1")
                raise e
        finally:
            del queue, temp

    log.info("Using {0} cores".format(options.cores))

    hdu_index = options.hdu_index
    if hdu_index > 0:
        log.info("Using hdu index {0}".format(hdu_index))

    # create a beam object from user input
    if options.beam is not None:
        beam = options.beam
        if len(beam) != 3:
            beam = beam.split()
            print "Beam requires 3 args. You supplied '{0}'".format(beam)
            sys.exit(1)
        options.beam = Beam(beam[0], beam[1], beam[2])
        log.info("Using user supplied beam parameters")
        log.info("Beam is {0} deg x {1} deg with pa {2}".format(options.beam.a, options.beam.b, options.beam.pa))

    # determine the latitude of the telescope
    if options.telescope is not None:
        lat = scope2lat(options.telescope)
    elif options.lat is not None:
        lat = options.lat
    else:
        lat = None

    # Generate and save the background FITS files
    if options.save:
        save_background_files(filename, hdu_index=hdu_index, cores=options.cores, beam=options.beam,
                              outbase=options.outbase)

    # auto-load background, noise, psf and region files
    if options.autoload:
        basename = os.path.splitext(filename)[0]
        if os.path.exists(basename+'_bkg.fits'):
            options.backgroundimg = basename+'_bkg.fits'
            log.info("Found background {0}".format(options.backgroundimg))
        if os.path.exists(basename+"_rms.fits"):
            options.noiseimg = basename+'_rms.fits'
            log.info("Found noise {0}".format(options.noiseimg))
        if os.path.exists(basename+".mim"):
            options.region = basename+".mim"
            log.info("Found region {0}".format(options.region))
        if os.path.exists(basename+"_psf.fits"):
            options.imgpsf = basename +"_psf.fits"
            log.info("Found psf {0}".format(options.imgpsf))

    # check that the aux input files exist
    if options.backgroundimg and not os.path.exists(options.backgroundimg):
        log.error("{0} not found".format(options.backgroundimg))
        sys.exit(1)
    if options.noiseimg and not os.path.exists(options.noiseimg):
        log.error("{0} not found".format(options.noiseimg))
        sys.exit(1)
    if options.imgpsf and not os.path.exists(options.imgpsf):
        log.error("{0} not found".format(options.imgpsf))
        sys.exit(1)
    if options.catpsf and not os.path.exists(options.catpsf):
        log.error("{0} not found".format(options.catpsf))
        sys.exit(1)

    if options.region is not None:
        if not os.path.exists(options.region):
            log.error("Region file {0} not found")
            sys.exit(1)
        if not region_available:
            log.error("Could not import AegeanTools/Region.py")
            log.error("(you probably need to install HealPy)")
            sys.exit(1)

    # check that the output table formats are supported (if given)
    # BEFORE any cpu intensive work is done
    if options.tables is not None:
        check_table_formats(options.tables)

    # if an outputfile was specified open it for writing, otherwise use stdout
    if not options.outfile:
        options.outfile = sys.stdout
    else:
        options.outfile = open(options.outfile, 'w')

    sources = []

    # do forced measurements using catfile
    if options.measure and options.priorized==0:
        if options.input is None:
            log.error("Must specify input catalog when --measure is selected")
            sys.exit(1)
        if not os.path.exists(options.input):
            log.error("{0} not found".format(options.input))
            sys.exit(1)
        log.info("Measuring fluxes of input catalog.")
        measurements = measure_catalog_fluxes(filename, catfile=options.input, hdu_index=options.hdu_index,
                                              outfile=options.outfile, bkgin=options.backgroundimg,
                                              rmsin=options.noiseimg, beam=options.beam, lat=lat)
        if len(measurements) == 0:
            log.info("No measurements made")
        sources.extend(measurements)

    if options.priorized>0:
        if options.ratio is not None:
            if options.ratio<=0:
                log.error("ratio must be positive definite")
                sys.exit(1)
            if options.ratio<1:
                log.error("ratio <1 is not advised. Have fun!")
        if options.input is None:
            log.error("Must specify input catalog when --priorized is selected")
            sys.exit(1)
        if not os.path.exists(options.input):
            log.error("{0} not found".format(options.input))
            sys.exit(1)
        log.info("Priorized fitting of sources in input catalog.")

        log.info("Stage = {0}".format(options.priorized))
        if options.doislandflux:
            log.warn("--island requested but not yet supported for priorized fitting")
        measurements = priorized_fit_islands(filename, catfile=options.input, hdu_index=options.hdu_index,
                                            rms=options.rms,
                                            outfile=options.outfile, bkgin=options.backgroundimg,
                                            rmsin=options.noiseimg, beam=options.beam, lat=lat, imgpsf=options.imgpsf,
                                            catpsf=options.catpsf,
                                            stage=options.priorized, ratio=options.ratio, outerclip=options.outerclip,
                                            cores=options.cores, doregroup=options.regroup)
        sources.extend(measurements)


    if options.find:
        log.info("Finding sources.")
        detections = find_sources_in_image(filename, outfile=options.outfile, hdu_index=options.hdu_index,
                                           rms=options.rms,
                                           max_summits=options.max_summits,
                                           innerclip=options.innerclip,
                                           outerclip=options.outerclip, cores=options.cores, rmsin=options.noiseimg,
                                           bkgin=options.backgroundimg, beam=options.beam,
                                           doislandflux=options.doislandflux,
                                           nonegative=not options.negative, nopositive=options.nopositive,
                                           mask=options.region, lat=lat, imgpsf=options.imgpsf)
        if len(detections) == 0:
            log.info("No sources found in image")
        sources.extend(detections)

    if len(sources) > 0 and options.tables:
        meta = {"PROGRAM":"Aegean",
                "PROGVER":"{0}-({1})".format(__version__,__date__),
                "FITSFILE":filename}
        for t in options.tables.split(','):
            save_catalog(t, sources)
    sys.exit()
