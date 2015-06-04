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

# logging and nice options
import logging
import logging.config
from optparse import OptionParser

# external and support programs
from AegeanTools.wcs_helpers import WCSHelper, PSFHelper
from AegeanTools.fits_image import FitsImage, Beam
from AegeanTools.msq2 import MarchingSquares
from AegeanTools.angle_tools import dec2hms, dec2dms, gcd, bear, translate
import AegeanTools.flags as flags
from AegeanTools.catalogs import show_formats, check_table_formats, load_table, load_catalog, table_to_source_list, save_catalog

from AegeanTools.models import OutputSource, IslandSource, SimpleSource, classify_catalog

#multiple cores support
import AegeanTools.pprocess as pprocess
import multiprocessing

__author__ = 'Paul Hancock'

# Aegean version [Updated via script]
__version__ = 'v1.9.6-0-g3df8610'
__date__ = '2015-06-03'

header = """#Aegean version {0}
# on dataset: {1}"""

#global constants
fwhm2cc = 1 / (2 * math.sqrt(2 * math.log(2)))
cc2fwhm = (2 * math.sqrt(2 * math.log(2)))

####################################### CLASSES ####################################
class GlobalFittingData(object):
    """
    The global data used for fitting.
    (should be) Read-only once created.
    Used by island fitting subprocesses.
    wcs parameter used by most functions.
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


class DummyMP(object):
    """
    A dummy copy of the mpfit class that just holds the parinfo variables
    This class doesn't do a great deal, but it makes it 'looks' like the mpfit class
    and makes it easy to estimate source parameters when you don't want to do any fitting.
    """

    def __init__(self, parinfo, perror):
        self.params = []
        for var in parinfo:
            try:
                val = var['value'][0]
            except:
                val = var['value']
            self.params.append(val)
        self.perror = perror
        self.errmsg = "There is no error, I just didn't bother fitting anything!"


class DummyLM(object):
    """
    A dummy copy of the mpfit class that just holds the parinfo variables
    This class doesn't do a great deal, but it makes it 'looks' like the mpfit class
    and makes it easy to estimate source parameters when you don't want to do any fitting.
    """

    def __init__(self):
        self.residual = [np.nan,np.nan]
        self.success = False


######################################### FUNCTIONS ###############################

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
            yield data_box, xmin, xmax-1, ymin, ymax-1


##parameter estimates
def estimate_lmfit_parinfo(data, rmsimg, curve, beam, innerclip, outerclip=None, offsets=(0, 0), max_summits=None):
    """Estimates the number of sources in an island and returns initial parameters for the fit as well as
    limits on those parameters.

    input:
    data   - np.ndarray of flux values
    rmsimg - np.ndarray of 1sigma values
    curve  - np.ndarray of curvature values
    beam   - beam object
    innerclip - the inner clipping level for flux data, in sigmas
    offsets - the (x,y) offset of data within it's parent image
              this is required for proper WCS conversions

    returns: an instance of lmfit.Parameters()
    """
    debug_on = logging.getLogger('Aegean').isEnabledFor(logging.DEBUG)
    is_flag = 0

    #is this a negative island?
    isnegative = max(data[np.where(np.isfinite(data))]) < 0
    if isnegative and debug_on:
        log.debug("[is a negative island]")

    #TODO: remove this later.
    if outerclip is None:
        outerclip = innerclip

    if debug_on:
        log.debug(" - shape {0}".format(data.shape))
        log.debug(" - xo_lim,yo_lim {0}, {1}".format(xo_lim, yo_lim))

    if not data.shape == curve.shape:
        log.error("data and curvature are mismatched")
        log.error("data:{0} curve:{1}".format(data.shape, curve.shape))
        raise AssertionError()

    #For small islands we can't do a 6 param fit
    #Don't count the NaN values as part of the island
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
        #1d islands or small islands only get one source
        if debug_on:
            log.debug("Tiny summit detected")
            log.debug("{0}".format(data))
        summits = [[data, 0, data.shape[0], 0, data.shape[1]]]
        #and are constrained to be point sources
        is_flag |= flags.FIXED2PSF
    else:
        if isnegative:
            #the summit should be able to include all pixels within the island not just those above innerclip
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
        # for writing ds9 style regions, prepend with "image"
        #print "point({0}, {1}) # point=x".format(int(offsets[1]),int(offsets[0]))

    # add summits in reverse order of peak SNR
    for summit, xmin, xmax, ymin, ymax in sorted(summits, key=lambda x: np.nanmax(-1.*abs(x[0]))):
        summits_considered += 1
        summit_flag = is_flag
        if debug_on:
            log.debug("Summit({5}) - shape:{0} x:[{1}-{2}] y:[{3}-{4}]".format(summit.shape, ymin, ymax, xmin, xmax, i))
        try:
            if isnegative:
                amp = np.nanmin(summit)
                xpeak, ypeak = np.unravel_index(np.nanargmin(summit),summit.shape)
            else:
                amp = np.nanmax(summit)
                xpeak, ypeak = np.unravel_index(np.nanargmax(summit),summit.shape)
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
        snr = np.nanmax(abs(data[xmin:xmax+1,ymin:ymax+1] / rmsimg[xmin:xmax+1,ymin:ymax+1]))
        if snr < innerclip:
            log.debug("Summit has SNR {0} < innerclip {1}: skipping".format(snr,innerclip))
            #log.warn("Summit doesn't obey inner clip constraint. This should have been caught earlier.")
            #log.info("snr {0} < innerclip {1}".format(snr,innerclip))
            #log.info("{0}".format(data[xmin:xmax+1,ymin:ymax+1] / rmsimg[xmin:xmax+1,ymin:ymax+1]))
            continue


        # allow amp to be 5% or (innerclip) sigma higher
        # TODO: the 5% should depend on the beam sampling
        # when innerclip is 400 this becomes rather stupid
        if amp > 0:
            amp_min, amp_max = 0.95 * min(outerclip * rmsimg[xo, yo], amp), amp * 1.05 + innerclip * rmsimg[xo, yo]
        else:
            amp_max, amp_min = 0.95 * max(-outerclip * rmsimg[xo, yo], amp), amp * 1.05 - innerclip * rmsimg[xo, yo]

        if debug_on:
            log.debug("a_min {0}, a_max {1}".format(amp_min, amp_max))


        #pixbeam = global_data.wcshelper.get_pixbeam_pixel(yo+offsets[0],xo+offsets[1])
        pixbeam = global_data.psfhelper.get_pixbeam_pixel(yo+offsets[0],xo+offsets[1])
        if pixbeam is None:
            if debug_on:
                log.debug("WCSERR")
            is_flag = flags.WCSERR
            pixbeam = Beam(1, 1, 0)

        #set a square limit based on the size of the pixbeam
        xo_lim = 0.5*np.hypot(pixbeam.a, pixbeam.b)
        yo_lim = xo_lim

        yo_min, yo_max = max(ymin, yo - yo_lim), min(ymax, yo + yo_lim)
        if yo_min == yo_max:  #if we have a 1d summit then allow the position to vary by +/-0.5pix
            yo_min, yo_max = yo_min - 0.5, yo_max + 0.5

        xo_min, xo_max = max(xmin, xo - xo_lim), min(xmax, xo + xo_lim)
        if xo_min == xo_max:  #if we have a 1d summit then allow the position to vary by +/-0.5pix
            xo_min, xo_max = xo_min - 0.5, xo_max + 0.5

        #TODO: The limits on sx,sy work well for circular beams or unresolved sources
        #for elliptical beams *and* resolved sources this isn't good and should be redone

        xsize = xmax - xmin + 1
        ysize = ymax - ymin + 1

        #initial shape is the pix beam
        sx = pixbeam.a * fwhm2cc
        sy = pixbeam.b * fwhm2cc

        # lmfit does silly things if we start with these two parameters being equal
        sx = max(sx,sy*1.01)

        #constraints are based on the shape of the island
        sx_min, sx_max = sx * 0.8, max((max(xsize, ysize) + 1) * math.sqrt(2) * fwhm2cc, sx * 1.1)
        sy_min, sy_max = sy * 0.8, max((max(xsize, ysize) + 1) * math.sqrt(2) * fwhm2cc, sx * 1.1)

        #TODO: update this to fit a psf for things that are "close" to a psf.
        #if the min/max of either sx,sy are equal then use a PSF fit
        if sy_min == sy_max or sx_min == sx_max: # this will never happen
            summit_flag |= flags.FIXED2PSF

        theta = pixbeam.pa # Degrees
        flag = summit_flag

        #check to see if we are going to fit this source
        if max_summits is not None:
            maxxed = i >= max_summits
        else:
            maxxed = False

        if maxxed:
            summit_flag |= flags.NOTFIT
            summit_flag |= flags.FIXED2PSF

        if debug_on:
            log.debug(" - var val min max | min max")
            log.debug(" - amp {0} {1} {2} ".format(amp, amp_min, amp_max))
            log.debug(" - xo {0} {1} {2} ".format(xo, xo_min, xo_max))
            log.debug(" - yo {0} {1} {2} ".format(yo, yo_min, yo_max))
            log.debug(" - sx {0} {1} {2} | {3} {4}".format(sx, sx_min, sx_max, sx_min * cc2fwhm,
                                                                  sx_max * cc2fwhm))
            log.debug(" - sy {0} {1} {2} | {3} {4}".format(sy, sy_min, sy_max, sy_min * cc2fwhm,
                                                                  sy_max * cc2fwhm))
            log.debug(" - theta {0} {1} {2}".format(theta, -180, 180))
            log.debug(" - flags {0}".format(flag))
            log.debug(" - fit?  {0}".format(not maxxed))

        # adjust parameters so that no two have the same initial value
        # TODO: incorporate the circular constraint
        prefix = "c{0}_".format(i)
        params.add(prefix+'amp',value=amp, min=amp_min, max=amp_max, vary= not maxxed)
        params.add(prefix+'xo',value=xo, min=float(xo_min), max=float(xo_max), vary= not maxxed)
        #params.add(prefix+'xo',value=xo, vary= not maxxed)
        params.add(prefix+'yo',value=yo, min=float(yo_min), max=float(yo_max), vary= not maxxed)
        #params.add(prefix+'yo',value=yo, vary= not maxxed)
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
    params.components=i
    if params.components <1:
        log.debug("Considered {0} summits, accepted {1}".format(summits_considered,i))
    return params


pass # for code folding purposes only!
# Modelling and fitting functions (now in AegeanTools.fitting)
# def elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta):
# def Cmatrix(x,y,sx,sy,theta):
# def Bmatrix(C):
# def emp_jacobian(pars, x, y, errs=None):
# def CRB_errs(jac, C, B=None):
# def ntwodgaussian_lmfit(params):
# def do_lmfit(data, params):
# def calc_errors(source, thetaN=None):

def result_to_components(result, model, island_data, isflags):
    """
    Convert fitting results into a set of components
    :return: a list of components
    """

    global global_data

    # island data
    isle_num = island_data.isle_num
    idata = island_data.i
    if island_data.scalars is not None:
        innerclip, outerclip, max_summits = island_data.scalars
    xmin, xmax, ymin, ymax = island_data.offsets


    rms = global_data.rmsimg[xmin:xmax + 1, ymin:ymax + 1]
    bkg = global_data.bkgimg[xmin:xmax + 1, ymin:ymax + 1]
    residual = np.median(result.residual),np.std(result.residual)
    is_flag = isflags

    sources = []
    for j in range(model.components):
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
        # Clamp the pixel location to the edge of the background map (see Trac #51)
        y = max(min(int(round(y_pix - ymin)), bkg.shape[1] - 1), 0)
        x = max(min(int(round(x_pix - xmin)), bkg.shape[0] - 1), 0)
        source.background = bkg[x, y]
        source.local_rms = rms[x, y]
        source.peak_flux = amp

        # position and shape
        if sx >= sy:
            major = sx
            axial_ratio = sx/sy #abs(sx*global_data.img.pixscale[0] / (sy * global_data.img.pixscale[1]))
        else:
            major = sy
            axial_ratio = sy/sx #abs(sy*global_data.img.pixscale[1] / (sx * global_data.img.pixscale[0]))
            theta = theta - 90

        # source.pa is returned in degrees
        (source.ra, source.dec, source.a, source.pa) = global_data.wcshelper.pix2sky_vec((x_pix, y_pix), major * cc2fwhm, theta)
        source.a *= 3600  # arcseconds
        source.b = source.a  / axial_ratio
        source.pa = pa_limit(source.pa)
        #fix_shape(source)

        # if one of these values are nan then there has been some problem with the WCS handling
        if not all(np.isfinite([source.ra, source.dec, source.a, source.pa])):
            src_flags |= flags.WCSERR
        # negative degrees is valid for RA, but I don't want them.
        if source.ra < 0:
            source.ra += 360
        source.ra_str = dec2hms(source.ra)
        source.dec_str = dec2dms(source.dec)



        # calculate integrated flux
        source.int_flux = source.peak_flux * sx * sy * cc2fwhm ** 2 * np.pi
        source.int_flux /= global_data.psfhelper.get_beamarea_pix(source.ra,source.dec) # scale Jy/beam -> Jy

        # Calculate errors for params that were fit (and int_flux)
        errors(source, model, global_data.wcshelper)

        source.flags = src_flags
        sources.append(source)
        log.debug(source)

    # calculate the integrated island flux if required
    if island_data.doislandflux:
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
        positions = np.where(kappa_sigma == source.peak_flux)
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

        # calculate the area of the island as a fraction of the area of the bounding box
        bl = global_data.wcshelper.pix2sky([xmax, ymin])
        tl = global_data.wcshelper.pix2sky([xmax, ymax])
        tr = global_data.wcshelper.pix2sky([xmin, ymax])
        height = gcd(tl[0], tl[1], bl[0], bl[1])
        width = gcd(tl[0], tl[1], tr[0], tr[1])
        area = height * width
        source.area = area * source.pixels / source.x_width / source.y_width

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
        beam_area = global_data.wcshelper.get_beamarea_pix(source.ra,source.dec)
        isize = source.pixels  #number of non zero pixels
        log.debug("- pixels used {0}".format(isize))
        source.int_flux = np.nansum(kappa_sigma)  #total flux Jy/beam
        log.debug("- sum of pixles {0}".format(source.int_flux))
        source.int_flux /= beam_area
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
    image = main image object
    auxfile = filename of auxiliary file
    """
    auximg = FitsImage(auxfile, beam=global_data.beam).get_pixels()
    if auximg.shape != image.get_pixels().shape:
        log.error("file {0} is not the same size as the image map".format(auxfile))
        log.error("{0}= {1}, image = {2}".format(auxfile, auximg.shape, image.get_pixels().shape))
        sys.exit()
    return auximg


def load_bkg_rms_image(image, bkgfile, rmsfile):
    """
    Load the background and rms images.
    Deprecation iminent:
      use load_aux_image on individual images
    """
    bkgimg = load_aux_image(image, bkgfile)
    rmsimg = load_aux_image(image, rmsfile)
    return bkgimg, rmsimg


def load_globals(filename, hdu_index=0, bkgin=None, rmsin=None, beam=None, verb=False, rms=None, cores=1, csigma=None,
                 do_curve=True, mask=None, lat=None, psf=None):
    """
    populate the global_data object by loading or calculating the various components
    """
    global global_data

    img = FitsImage(filename, hdu_index=hdu_index, beam=beam)
    beam = img.beam

    # Save global data for use by fitting sub-processes
    global_data = GlobalFittingData()

    debug = logging.getLogger('Aegean').isEnabledFor(logging.DEBUG)

    if mask is not None and region_available:
        # allow users to supply and object instead of a filename
        if isinstance(mask,Region):
            global_data.region = mask
        elif os.path.exists(mask):
            global_data.region = pickle.load(open(mask))
        else:
            log.error("File {0} not found for loading".format(mask))
    else:
        global_data.region = None

    #global_data.beam = beam
    #global_data.hdu_header = img.get_hdu_header()
    #global_data.wcs = img.wcs
    global_data.wcshelper = WCSHelper.from_header(img.get_hdu_header(), beam, lat)
    global_data.psfhelper = PSFHelper(psf, global_data.wcshelper)
    global_data.beam = global_data.wcshelper.beam
    #initial values of the three images
    global_data.img = img
    global_data.data_pix = img.get_pixels()
    global_data.dtype = type(global_data.data_pix[0][0])
    global_data.bkgimg = np.zeros(global_data.data_pix.shape, dtype=global_data.dtype)
    global_data.rmsimg = np.zeros(global_data.data_pix.shape, dtype=global_data.dtype)
    global_data.pixarea = img.pixarea
    global_data.dcurve = None
    if do_curve:
        #calculate curvature but store it as -1,0,+1
        cimg = curvature(global_data.data_pix)
        if csigma is None:
            log.info("Calculating curvature csigma")
            _, csigma = estimate_bkg_rms(cimg)
        dcurve = np.zeros(global_data.data_pix.shape, dtype=np.int8)
        dcurve[np.where(cimg <= -abs(csigma))] = -1
        dcurve[np.where(cimg >= abs(csigma))] = 1
        del cimg

        global_data.dcurve = dcurve

    #calculate the pixel beam
    #global_data.pixbeam = global_data.wcshelper.get_pixbeam()
    #log.debug("pixbeam is : {0}".format(global_data.pixbeam))

    #if either of rms or bkg images are not supplied then calculate them both
    if not (rmsin and bkgin):
        if verb:
            log.info("Calculating background and rms data")
        make_bkg_rms_from_global(mesh_size=20, forced_rms=rms, cores=cores)

    #if a forced rms was supplied use that instead
    if rms is not None:
        global_data.rmsimg = np.ones(global_data.data_pix.shape) * rms

    #replace the calculated images with input versions, if the user has supplied them.
    if bkgin:
        if verb:
            log.info("Loading background data from file {0}".format(bkgin))
        global_data.bkgimg = load_aux_image(img, bkgin)
    if rmsin:
        if verb:
            log.info("Loading rms data from file {0}".format(rmsin))
        global_data.rmsimg = load_aux_image(img, rmsin)

    #subtract the background image from the data image and save
    if verb and debug:
        log.debug("Data max is {0}".format(img.get_pixels()[np.isfinite(img.get_pixels())].max()))
        log.debug("Doing background subtraction")
    img.set_pixels(img.get_pixels() - global_data.bkgimg)
    global_data.data_pix = img.get_pixels()
    if verb and debug:
        log.debug("Data max is {0}".format(img.get_pixels()[np.isfinite(img.get_pixels())].max()))

    return

#image manipulation
def make_bkg_rms_image(data, beam, mesh_size=20, forced_rms=None):
    """
    [legacy version used by the pipeline]

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
        sys.exit()

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

    inputs:
    mesh_size - number of beams per box
                default = 20
    forced_rms - the rms of the image
                None => calculate the rms and bkg levels (default)
                <float> => assume zero background and constant rms
    cores - the maximum number of cores to use when multiprocessing
            default/None = One core only.
    return:
    nothing
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

    #calculate a local beam from the center of the data
    #pixbeam = global_data.wcshelper.get_pixbeam(xcen,ycen)
    pixbeam = global_data.psfhelper.get_pixbeam_pixel(xcen,ycen)
    if pixbeam is None:
        log.error("Cannot calculate the beam shape at the image center")
        sys.exit()

    width_x = mesh_size * max(abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.a),
                              abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.b))
    width_x = int(width_x)
    width_y = mesh_size * max(abs(math.sin(np.radians(pixbeam.pa)) * pixbeam.a),
                              abs(math.cos(np.radians(pixbeam.pa)) * pixbeam.b))
    width_y = int(width_y)

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

    if cores > 1:
        #set up the queue
        queue = pprocess.Queue(limit=cores, reuse=1)
        estimate = queue.manage(pprocess.MakeReusable(estimate_background_global))
        #populate the queue
        for xmin, xmax in zip(xmins, xmaxs):
            for ymin, ymax in zip(ymins, ymaxs):
                estimate(ymin, ymax, xmin, xmax)
    else:
        queue = []
        for xmin, xmax in zip(xmins, xmaxs):
            for ymin, ymax in zip(ymins, ymaxs):
                queue.append(estimate_background_global(xmin, xmax, ymin, ymax))

    #construct the bkg and rms images
    if global_data.rmsimg is None:
        global_data.rmsimg = np.zeros(data.shape, dtype=global_data.dtype)
    if global_data.bkgimg is None:
        global_data.bkgimg = np.zeros(data.shape, dtype=global_data.dtype)

    for ymin, ymax, xmin, xmax, bkg, rms in queue:
        global_data.bkgimg[ymin:ymax, xmin:xmax] = bkg
        global_data.rmsimg[ymin:ymax, xmin:xmax] = rms
    return


def estimate_background_global(xmin, xmax, ymin, ymax):
    """
    Estimate the background noise mean and RMS.
    The mean is estimated as the median of data.
    The RMS is estimated as the IQR of data / 1.34896.
    Returns nothing
    reads/writes data from global_data
    works only on the sub-region specified by
    ymin,ymax,xmin,xmax
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
    #return the input and output data so we know what we are doing
    # when compiling the results of multiple processes
    return ymin, ymax, xmin, xmax, bkg, rms


def estimate_bkg_rms(data):
    """
    Estimate the background noise mean and RMS.
    The mean is estimated as the median of data.
    The RMS is estimated as the IQR of data / 1.34896.
    Returns (bkg, rms).
    Returns (NaN, NaN) if data contains fewer than 4 values.
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


def curvature(data, aspect=None):
    """
    Use a Laplacian kernel to calculate the curvature map.
    input:
        data - the image data
        aspect - the ratio of pixel size (x/y)
                NOT TESTED!
    """
    if not aspect:
        kern = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    else:
        log.warn("Aspect != has not been tested.")
        #TODO: test that this actually works as intended
        a = 1.0 / aspect
        b = 1.0 / math.sqrt(1 + aspect ** 2)
        c = -2.0 * (1 + a + 2 * b)
        kern = 0.25 * np.array([[b, a, b], [1, c, 1], [b, a, b]])
    return ndi.convolve(data, kern)


##Nifty helpers
def within(x, xm, xx):
    """Enforce xm<= x <=xx"""
    return min(max(x, xm), xx)


def fix_shape(source):
    """
    Ensure that a>=b
    adjust as required
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
    """
    while pa <= -90:
        pa += 180
    while pa > 90:
        pa -= 180
    return pa


def theta_limit(theta):
    """
    Position angle is periodic with period 180\deg
    Constrain pa such that -pi/2<theta<=pi/2
    """
    while theta <= -1*np.pi/2:
        theta += np.pi
    while theta > np.pi/2:
        theta -= np.pi
    return theta


def gmean(indata):
    """
    Calculate the geometric mean of a data set taking account of
    values that may be negative, zero, or nan
    :param data: a list of floats or ints
    :return: the geometric mean of the data
    """
    data = np.ravel(indata)
    if np.inf in data:
        return np.inf, np.inf

    finite = data[np.isfinite(data)]
    if len(finite) < 1:
        return np.nan, np.nan
    #determine the zero point and scale all values to be 1 or greater
    scale = abs(np.min(finite)) + 1
    finite += scale
    #calculate the geometric mean of the scaled data and scale back
    lfinite = np.log(finite)
    flux = np.exp(np.mean(lfinite)) - scale
    error = np.nanstd(lfinite) * flux
    return flux, abs(error)


#WCS helper functions
def scope2lat(telescope):
    """
    Convert a telescope name into a latitude
    :param telescope:
    :return:
    """
    # TODO: look at http://en.wikipedia.org/wiki/List_of_radio_telescopes
    # and add some more telescopes to this list
    if telescope.lower() == 'mwa':
        return -26.703319 # Hopefully wikipedia is correct
    elif telescope.lower() == 'atca':
        return -30.3128 # From google
    elif telescope.lower() == 'vla':
        return 34.0790 # From google
    elif telescope.lower() == 'lofar':
        return 52.9088 # From google
    else:
        log.warn("Telescope {0} is unknown".format(telescope))
        log.warn("integrated fluxes may be incorrect")
        return None

pass # The following have been moved into AegeanTools.wcs_helpers
# and can now be accessed via global_data.wcshelper.<func>
# def pix2sky(pixel):
# def sky2pix(pos):
# def sky2pix_vec(pos, r, pa):
# def pix2sky_vec(pixel, r, theta):
# def get_pixbeam():
# def get_beamarea_deg2(ra,dec):
# def get_beamarea_pix(ra,dec):
# def sky_sep(pix1, pix2):

######################################### THE MAIN DRIVING FUNCTIONS ###############

#source finding and fitting
def refit_islands(group, stage, outerclip, istart):
    """
    Do island refitting (priorized fitting) on a group of islands.
    This function is to be driven by code that uses 1 or more cores.
    :param group: A list of islands group=[ [(0,0),(0,1)],[(1,0)] ...]
    :param stage: refit stage
    :param outerclip: outerclip for floodfill. outerclip<0 means no no clipping
    :param istart: the starting island number
    :return:
    """

    sources = []

    data = global_data.data_pix
    rmsimg = global_data.rmsimg
    #pixbeam = global_data.wcshelper.get_pixbeam()

    for inum, isle in enumerate(group,start=istart):
        log.debug("-=-")
        log.debug("input island = {0}, {1} components".format(isle[0].island, len(isle)))

        # set up the parameters for each of the sources within the island
        i = 0
        params = lmfit.Parameters()
        shape = data.shape
        xmin, ymin = shape
        xmax = ymax = 0

        island_mask = []
        for src in isle:
            #pixbeam = global_data.wcshelper.get_pixbeam(src.ra,src.dec)
            pixbeam = global_data.psfhelper.get_pixbeam(src.ra, src.dec)
            # find the right pixels from the ra/dec
            source_x, source_y = global_data.wcshelper.sky2pix([src.ra, src.dec])
            source_x -=1
            source_y -=1
            x = int(round(source_x))
            y = int(round(source_y))

            log.debug("pixel location ({0:5.2f},{1:5.2f})".format(source_x,source_y))
            # reject sources that are outside the image bounds, or which have nan data/rms values
            if not 0 <= x < shape[0] or not 0 <= y < shape[1] or \
                    not np.isfinite(data[x, y]) or \
                    not np.isfinite(rmsimg[x, y]):
                log.info("Source ({0},{1}) not within usable region: skipping".format(src.island,src.source))
                continue
            # determine the shape parameters in pixel values
            sx, theta = global_data.wcshelper.sky2pix_vec([src.ra,src.dec], src.a/3600., src.pa)[2:]
            # we have to keep the PA the same here otherwise our shape parameters are incorrect
            # this may be compensating for an error we made elsewhere but for now it works.
            sy = global_data.wcshelper.sky2pix_vec([src.ra,src.dec], src.b/3600., src.pa)[2]
            sx *=fwhm2cc
            sy *=fwhm2cc

            log.debug("Source shape [sky coords]  {0:5.2f}x{1:5.2f}@{2:05.2f}".format(src.a,src.b,src.pa))
            log.debug("Source shape [pixel coords] {0:4.2f}x{1:4.2f}@{2:05.2f}".format(sx,sy,theta))

            # choose a region that is 2x the major axis of the source, 4x semimajor axis a
            width = 2 * sx
            ywidth = int(round(width)) + 1
            xwidth = int(round(width)) + 1

            # adjust the size of the island to include this source
            xmin = min(xmin, max(0, x - xwidth / 2))
            ymin = min(ymin, max(0, y - ywidth / 2))
            xmax = max(xmax, min(shape[0], x + xwidth / 2 + 1))
            ymax = max(ymax, min(shape[1], y + ywidth / 2 + 1))

            s_lims = [0.8 * min(sx,pixbeam.b * fwhm2cc), 2 * sy * math.sqrt(2)]

            # Set up the parameters for the fit, including constraints
            prefix = "c{0}_".format(i)
            params.add(prefix + 'amp', value=src.peak_flux*2) # always vary
            # for now the xo/yo are locations within the main image, we correct this later
            params.add(prefix + 'xo', value=source_x, min=source_x-sx/2., max=source_x+sx/2., vary= stage>=2)
            params.add(prefix + 'yo', value=source_y, min=source_y-sy/2., max=source_y+sy/2., vary= stage>=2)
            params.add(prefix + 'sx', value=sx, min=s_lims[0], max=s_lims[1], vary= stage>=3)
            params.add(prefix + 'sy', value=sy, min=s_lims[0], max=s_lims[1], vary= stage>=3)
            params.add(prefix + 'theta', value=theta, vary= stage>=3)
            params.add(prefix + 'flags', value=0, vary=False)
            i += 1

            # Use pixels above outerclip sigmas..
            if outerclip>=0:
                mask = np.where(data[xmin:xmax,ymin:ymax]-outerclip*rmsimg[xmin:xmax,ymin:ymax]>0)
            else: # negative outer clip means not clipping at all
                mask = np.where(data[xmin:xmax,ymin:ymax])

            # convert the pixel indices to be pixels within the parent data set
            xmask = mask[0] + xmin
            ymask = mask[1] + ymin
            island_mask.extend(zip(xmask,ymask))

        if i==0:
            log.info("No sources found in island {0}".format(src.island))
            continue
        params.components = i
        log.debug(" {0} components being fit".format(i))
        # now we correct the xo/yo positions to be relative to the sub-image
        log.debug("xmxxymyx {0} {1} {2} {3}".format(xmin,xmax,ymin,ymax))
        for i in range(params.components):
            try:
                prefix = "c{0}_".format(i)
                params[prefix + 'xo'].value -=xmin
                params[prefix + 'xo'].min -=xmin
                params[prefix + 'xo'].max -=xmin
                params[prefix + 'yo'].value -=ymin
                params[prefix + 'yo'].min -=ymin
                params[prefix + 'yo'].max -=ymin
            except Exception, e:
                log.error(" ARG !")
                log.info(params)
                log.info(params.components)
                log.info("trying to access component {0}".format(i))
                raise e
        log.debug(params)
        # don't fit if there are no sources
        if params.components<1:
            log.info("Island {0} has no components".format(src.island))
            continue

        # this .copy() will stop us from modifying the parent region when we later apply our mask.
        idata = data[xmin:xmax, ymin:ymax].copy()
        # now convert these back to indices within the idata region
        island_mask = [(x-xmin,y-ymin) for x,y in island_mask]
        # the mask is for good pixels so we need to reverse it
        all_pixels = zip(*np.where(idata))
        mask = zip(*set(all_pixels).difference(set(island_mask)))
        idata[mask] = np.nan # this is the mask mentioned above

        mx,my = np.where(np.isfinite(idata))
        non_nan_pix = len(mx)

        log.debug("island extracted:")
        log.debug(" x[{0}:{1}] y[{2}:{3}]".format(xmin,xmax,ymin,ymax))
        log.debug(" max = {0}".format(np.nanmax(idata)))
        log.debug(" total {0}, masked {1}, not masked {2}".format(len(all_pixels),non_nan_pix,len(all_pixels)-non_nan_pix))

        # determine the number of free parameters and if we have enough data for a fit
        nfree = np.count_nonzero([params[p].vary for p in params.keys()])
        if non_nan_pix < nfree:
            log.debug("More free parameters {0} than available pixels {1}".format(nfree,non_nan_pix))
            if non_nan_pix >= params.components:
                log.debug("Fixing all parameters except amplitudes")
                for p in params.keys():
                    if 'amp' not in p:
                        params[p].vary = False
            else:
                log.debug(" no not-masked pixels, skipping".format(src.island,src.source))
            continue

        # do the fit
        fac = 1/np.sqrt(2) # TODO: why sqrt(2)?
        C = Cmatrix(mx, my, pixbeam.a*fwhm2cc*fac, pixbeam.b*fwhm2cc*fac, pixbeam.pa)
        B = Bmatrix(C)
        errs = np.nanmax(rmsimg[xmin:xmax, ymin:ymax])
        result, model = do_lmfit(idata, params, B=B)
        model = covar_errors(model, idata, errs=errs, B=B, C=C)

        # convert the results to a source object
        offsets = (xmin, xmax, ymin, ymax)
        island_data = IslandFittingData(inum, offsets=offsets)
        new_src = result_to_components(result, model, island_data, src.flags)

        # preserve the uuid so we can do exact matching between catalogs
        for ns, s in zip(new_src,isle):
            ns.uuid = s.uuid
        sources.extend(new_src)
    return sources


def fit_island(island_data):
    """
    Take an Island and do all the parameter estimation and fitting.
    island_data - an IslandFittingData object
    Return a list of sources that are within the island.
    None = no sources found in the island.
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

    # isle = Island(idata)
    icurve = dcurve[xmin:xmax + 1, ymin:ymax + 1]
    rms = rmsimg[xmin:xmax + 1, ymin:ymax + 1]

    is_flag = 0
    #pixbeam = global_data.wcshelper.get_pixbeam_pixel((xmin+xmax)/2., (ymin+ymax)/2.)
    pixbeam = global_data.psfhelper.get_pixbeam_pixel((xmin+xmax)/2., (ymin+ymax)/2.)
    if pixbeam is None:
        is_flag |= flags.WCSERR

    log.debug("=====")
    log.debug("Island ({0})".format(isle_num))

    params = estimate_lmfit_parinfo(idata, rms, icurve, beam, innerclip, outerclip, offsets=[xmin, ymin],
                               max_summits=max_summits)

    # islands at the edge of a region of nans
    # they result in no components
    if params is None or params.components <1:
        return []

    log.debug("Rms is {0}".format(np.shape(rms)))
    log.debug("Isle is {0}".format(np.shape(idata)))
    log.debug(" of which {0} are masked".format(sum(np.isnan(idata).ravel() * 1)))

    # TODO: Allow for some of the components to be fit if there are multiple components in the island
    # Check that there is enough data to do the fit
    mx,my = np.where(np.isfinite(idata))
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
        fac = 1/np.sqrt(2) # TODO: why sqrt(2)?
        C = Cmatrix(mx, my, pixbeam.a*fwhm2cc*fac, pixbeam.b*fwhm2cc*fac, pixbeam.pa)
        B = Bmatrix(C)
        #B = None
        log.debug("C({0},{1},{2},{3},{4})".format(len(mx),len(my),pixbeam.a*fwhm2cc, pixbeam.b*fwhm2cc, pixbeam.pa))
        #snr = np.nanmax(idata)/np.nanmax(rms)
        errs = np.nanmax(rms)
        result, model = do_lmfit(idata, params, B=B)
        if not result.errorbars:
            is_flag |= flags.FITERR
        # get the real parameter errors
        model = covar_errors(model, idata, errs=errs, B=B, C=C)

        if not result.success:
            is_flag |= flags.FITERR

    log.debug(model)

    # convert the fitting results to a list of sources +/0 islands
    sources =result_to_components(result, model, island_data, is_flag)

    return sources


def fit_islands(islands):
    """
    Execute fitting on a list of islands.
      islands - a list of IslandFittingData objects
    Returns a list of OutputSources
    """
    log.debug("Fitting group of {0} islands".format(len(islands)))
    sources = []
    for island in islands:
        res = fit_island(island)
        sources.extend(res)
    return sources


def find_sources_in_image(filename, hdu_index=0, outfile=None, rms=None, max_summits=None, csigma=None, innerclip=5,
                          outerclip=4, cores=None, rmsin=None, bkgin=None, beam=None, doislandflux=False,
                          returnrms=False, nopositive=False, nonegative=False, mask=None, lat=None, psf=None):
    """
    Run the Aegean source finder.
    Inputs:
    filename    - the name of the input file (FITS only)
    hdu_index   - the index of the FITS HDU (extension)
                   Default = 0
    outfile     - print the resulting catalogue to this file
                   Default = None = don't print to a file
    rms         - use this rms for the entire image (will also assume that background is 0)
                   default = None = calculate rms and background values
    max_summits - ignore any islands with more summits than this
                   Default = None = no limit
    csigma      - use this as the clipping limit for the curvature map
                   Default = None = calculate the curvature rms and use 1sigma
    innerclip   - the seeding clip, in sigmas, for seeding islands of pixels
                   Default = 5
    outerclip   - the flood clip in sigmas, used for flooding islands
                   Default = 4
    cores       - number of CPU cores to use. None means all cores.
    rmsin       - filename of an rms image that will be used instead of
                   the internally calculated one
    bkgin       - filename of a background image that will be used instead of
                    the internally calculated one
    beam        - (major,minor,pa) (all degrees) of the synthesised beam to be use
                   overrides whatever is given in the FITS header.
    doislandflux- if true, an integrated flux will be returned for each island in addition to
                    the individual component entries.
    returnrms   - if true, also return the rms image. Default=False
    nopositive  - if true, sources with positive fluxes will not be reported
    nonegative  - if true, sources with negative fluxes will not be reported
    mask        - the filename of a region file created by MIMAS. Islands outside of this region will be ignored.

    Return:
    if returnrms:
        [ [OutputSource,...], rmsimg]
    else:
        [OuputSource,... ]
    """
    np.seterr(invalid='ignore')
    if cores is not None:
        assert (cores >= 1), "cores must be one or more"

    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, beam=beam, rms=rms, cores=cores,
                 csigma=csigma, verb=True, mask=mask, lat=lat, psf=psf)

    #we now work with the updated versions of the three images
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
        #This check is also made during start up when running aegean from the command line
        #However I reproduce it here so that we don't fall over when aegean is being imported
        #into other codes (eg the VAST pipeline)
        if cores is None:
            cores = multiprocessing.cpu_count()
            log.info("Found {0} cores".format(cores))
        else:
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
            #empty islands have length 1
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

    for src in queue:
        if src:  # ignore src==None
            #src is actually a list of sources
            for s in src:
                #ignore sources that we have been told to ignore
                if (s.peak_flux > 0 and nopositive) or (s.peak_flux < 0 and nonegative):
                    continue
                sources.append(s)
    if outfile:
        components, islands, simples = classify_catalog(sources)
        for source in sorted(components):
            outfile.write(str(source))
            outfile.write("\n")
    if returnrms:
        return [sources, global_data.rmsimg]
    else:
        return sources


def VASTP_find_sources_in_image():
    """
    A version of find_sources_in_image that will accept already open files from the VAST pipeline.
    Should have identical behaviour to the non-pipeline version.
    """
    pass


def priorized_fit_islands(filename, catfile, hdu_index=0, outfile=None, bkgin=None, rmsin=None, cores=1, rms=None,
                           beam=None, lat=None, psf=None, stage=3, ratio=1.0, outerclip=3, doregroup=True):
    """
    Take an input catalog, and image, and optional background/noise images
    fit the flux and ra/dec for each of the given sources, keeping the morphology fixed

    if doregroup is true the groups will be recreated based on a matching radius/probability.
    if doregroup is false then the islands of the input catalog will be preserved.

    Multiple cores can be specified, and will be used.

    :param filename: image file or hdu
    :param catfile: catalog file name
    :param hdu_index:
    :param outfile: ouput file for ascii output (not tables)
    :param bkgin: background image name
    :param rmsin: noise image name
    :param cores: number of cores to use
    :param rms: if not none, then use this constant rms for the entire image
    :param beam: beam parameters to be used instead of those that may be in the fits header
    :param lat: latitude of telescope
    :param stage: refitting stage
    :param ratio: ratio of image psf to catalog psf
    :param outerclip: pixels above an snr of this amuont will be used in fitting, <0 -> all pixels.
    :param doregroup:  True - doregroup, False - use island data for groups
    :return: a list of source objects
    """

    from AegeanTools.cluster import regroup
    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, verb=True,
                 do_curve=False, beam=beam, lat=lat)

    beam = global_data.beam
    far = 10*beam.a/3600
    # load the table and convert to an input source list
    input_table = load_table(catfile)
    input_sources = table_to_source_list(input_table)
    # the input sources are the intial conditions for our fits.
    # Expand each source size if needed.
    if ratio is not None:
        far *= ratio
        for src in input_sources:
            src.a = np.sqrt(src.a**2 + (beam.a*3600)**2*(1-1/ratio**2))
            src.b = np.sqrt(src.b**2 + (beam.b*3600)**2*(1-1/ratio**2))



    # redo the grouping if required
    if doregroup:
        #input_sources = sorted(input_sources, key = lambda x: x.dec)
        groups = regroup(input_sources, eps=np.sqrt(2), far=far)
    else:
        groups = list(island_itergen(input_sources))


    if cores == 1:  #single-threaded, no parallel processing
        queue = []
    else:
        #This check is also made during start up when running aegean from the command line
        #However I reproduce it here so that we don't fall over when aegean is being imported
        #into other codes (eg the VAST pipeline)
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

    for i,island in enumerate(groups):
        island_group.append(island)
        # If the island group is full queue it for the subprocesses to fit
        if len(island_group) >= group_size:
            if cores > 1:
                fit_parallel(island_group, stage, outerclip, istart =i)
            else:
                res = refit_islands(island_group, stage, outerclip, istart=i)
                queue.append(res)
            island_group = []

    # The last partially-filled island group also needs to be queued for fitting
    if len(island_group) > 0:
        if cores > 1:
            fit_parallel(island_group, stage, outerclip, istart =i)
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
    for source in sources:
        print >> outfile, str(source)

    print "found {0} sources".format(len(sources))
    return sources


def island_itergen(catalog):
    """
    Iterate over a catalog of sources, and return an island worth of sources at a time.
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


#just flux measuring
def force_measure_flux(radec):
    """
    Measure the flux of a point source at each of the specified locations
    Not fitting is done, just forced measurements
    Assumes that global_data hase been populated
    input:
        img - the image data [array]
        radec - the locations at which to measure fluxes
    returns:
    [(flux,err),...]
    """
    from AegeanTools.fitting import ntwodgaussian_mpfit
    catalog = []

    #this is what we use to denote sources that are we are not able to measure
    dummy = SimpleSource()
    dummy.peak_flux = np.nan
    dummy.peak_pixel = np.nan
    dummy.flags = flags.FITERR

    shape = global_data.data_pix.shape

    if global_data.wcshelper.lat is not None:
        log.warn("No account is being made for telescope latitude, even though it has been supplied")
    for ra, dec in radec:
        #find the right pixels from the ra/dec
        source_x, source_y = global_data.wcshelper.sky2pix([ra, dec])
        x = int(round(source_x))
        y = int(round(source_y))

        #reject sources that are outside the image bounds, or which have nan data/rms values
        if not 0 <= x < shape[0] or not 0 <= y < shape[1] or \
                not np.isfinite(global_data.data_pix[x, y]) or \
                not np.isfinite(global_data.rmsimg[x, y]):
            catalog.append(dummy)
            continue

        flag = 0
        #make a pixbeam at this location
        #pixbeam = global_data.wcshelper.get_pixbeam(ra,dec)
        pixbeam = global_data.psfhelper.get_pixbeam(ra,dec)
        if pixbeam is None:
            flag |= flags.WCSERR
            pixbeam = Beam(1, 1, 0)
        #determine the x and y extent of the beam
        xwidth = 2 * pixbeam.a * pixbeam.b
        xwidth /= np.hypot(pixbeam.b * np.sin(np.radians(pixbeam.pa)), pixbeam.a * np.cos(np.radians(pixbeam.pa)))
        ywidth = 2 * pixbeam.a * pixbeam.b
        ywidth /= np.hypot(pixbeam.b * np.cos(np.radians(pixbeam.pa)), pixbeam.a * np.sin(np.radians(pixbeam.pa)))
        #round to an int and add 1
        ywidth = int(round(ywidth)) + 1
        xwidth = int(round(xwidth)) + 1

        #cut out an image of this size
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
        params = [amp, xo, yo, pixbeam.a * fwhm2cc, pixbeam.b * fwhm2cc, pixbeam.pa]
        gaussian_data = ntwodgaussian_mpfit(params)(*np.indices(data.shape))

        # Calculate the "best fit" amplitude as the average of the implied amplitude
        # for each pixel. Error is stddev.
        # Only use pixels within the FWHM, ie value>=0.5. Set the others to NaN
        ratios = np.where(gaussian_data >= 0.5, data / gaussian_data, np.nan)
        flux, error = gmean(ratios)

        #sources with fluxes or flux errors that are not finite are not valid
        # an error of identically zero is also not valid.
        if not np.isfinite(flux) or not np.isfinite(error) or error == 0.0:
            catalog.append(dummy)
            continue

        source = SimpleSource()
        source.ra = ra
        source.dec = dec
        source.peak_flux = flux  #* isnegative
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

    Input:
        filename - fits image file name to be read
        catfile - a catalog of source positions (ra,dec)
        hdu_index - if fits file has more than one hdu, it can be specified here
        outfile - the output file to write to
        bkgin - a background image filename
        rmsin - an rms image filename
        cores - cores to use
        rms - forced rms value
        beam - beam parameters to override those given in fits header

    """
    load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, verb=True,
                 do_curve=False, beam=beam, lat=lat, psf=psf)

    #load catalog
    radec = load_catalog(catfile)
    #measure fluxes
    sources = force_measure_flux(radec)
    #write output
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
    #measure fluxes
    if debug:
        level = log.getLogger().getEffectiveLevel()
        log.getLogger().setLevel(log.DEBUG)
    sources = force_measure_flux(positions)
    if debug:
        log.getLogger().setLevel(level)
    return sources


#secondary capabilities
def save_background_files(image_filename, hdu_index=0, bkgin=None, rmsin=None, beam=None, rms=None, cores=1,
                          outbase=None):
    """
    Generate and save the background and RMS maps as FITS files.
    They are saved in the current directly as aegean-background.fits and aegean-rms.fits.
    """
    global global_data

    log.info("Saving background / RMS maps")
    #load image, and load/create background/rms images
    load_globals(image_filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, beam=beam, verb=True, rms=rms,
                 cores=cores, do_curve=True)
    img = global_data.img
    bkgimg, rmsimg = global_data.bkgimg, global_data.rmsimg
    curve = np.array(global_data.dcurve,dtype=np.float32)#curvature(global_data.data_pix)
    #mask these arrays the same as the data
    bkgimg[np.where(np.isnan(global_data.data_pix))] = np.NaN
    rmsimg[np.where(np.isnan(global_data.data_pix))] = np.NaN
    curve[np.where(np.isnan(global_data.data_pix))] = np.NaN

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

#command line version of this program runs from here.
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
                      help="A .fits file that represents the image noise (rms), created from Aegean with --save or BANE. [default: none]")
    parser.add_option('--background', dest='backgroundimg', default=None,
                      help="A .fits file that represents the background level, created from Aegean with --save or BANE. [default: none]")
    parser.add_option('--psf', dest='psfimg',default=None,
                      help="A .fits file that represents the size (degrees) of a blurring disk. This disk is convolved with the BMAJ/BMIN listed in the FITS header and the result becomes the local PSF.")
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
                      help='The latitude of the tlescope used to collect data.')
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

    (options, args) = parser.parse_args()


    # configure logging
    global log
    logging_level = logging.DEBUG if options.debug else logging.INFO
    config = {"level":logging_level, "format":"%(module)s:%(levelname)s %(message)s"}
    logging.basicConfig(**config)
    # set up logging for Aegean which other modules can join
    log = logging.getLogger("Aegean")
    log.info("This is Aegean {0}-({1})".format(__version__,__date__))


    if options.table_formats:
        show_formats()
        sys.exit(0)

    if options.file_versions:
        log.info("Numpy {0} from {1} ".format(np.__version__, np.__file__))
        log.info("Scipy {0} from {1}".format(scipy.__version__, scipy.__file__))
        log.info("AstroPy {0} from {1}".format(astropy.__version__, astropy.__file__))
        sys.exit()

    #print help if the user enters no options or filename
    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    #check that a valid filename was entered
    filename = args[0]
    if not os.path.exists(filename):
        log.error("{0} not found".format(filename))
        sys.exit(-1)

    # tell numpy to shut up about "invalid values encountered"
    # Its just NaN's and I don't need to hear about it once per core
    np.seterr(invalid='ignore', divide='ignore')

    #check for nopositive/negative conflict
    if options.nopositive and not options.negative:
        log.warning('Requested no positive sources, but no negative sources. Nothing to find.')
        sys.exit()

    #if measure/save are enabled we turn off "find" unless it was specifically
    if (options.measure or options.save or options.priorized) and not options.find:
        options.find = False
    else:
        options.find = True

    #debugging in multi core mode is very hard to understand
    if options.debug:
        log.info("Setting cores=1 for debugging")
        options.cores = 1

    #check/set cores to use
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
                cores = 1
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

    #create a beam object from user input
    if options.beam is not None:
        beam = options.beam
        if len(beam) != 3:
            beam = beam.split()
            print "Beam requires 3 args. You supplied '{0}'".format(beam)
            sys.exit()
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

    # autoload bakground, noise and regio files
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
            options.psfimg = basename +"_psf.fits"
            log.info("Found psf {0}".format(options.psfimg))


    #check that the background and noise files exist
    if options.backgroundimg and not os.path.exists(options.backgroundimg):
        log.error("{0} not found".format(options.backgroundimg))
        sys.exit()
    if options.noiseimg and not os.path.exists(options.noiseimg):
        log.error("{0} not found".format(options.noise))
        sys.exit()

    #check that the output table formats are supported (if given)
    # BEFORE any cpu intensive work is done
    if options.tables is not None:
        check_table_formats(options.tables)

    #if an outputfile was specified open it for writing, otherwise use stdout
    if not options.outfile:
        options.outfile = sys.stdout
    else:
        options.outfile = open(options.outfile, 'w')

    if options.region is not None:
        if not os.path.exists(options.region):
            log.error("Region file {0} not found")
            sys.exit()
        if not region_available:
            log.error("Could not import AegeanTools/Region.py")
            log.error("(you probably need to install HealPy)")
            sys.exit()

    #do forced measurements using catfile
    sources = []
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
        measurements = priorized_fit_islands(filename, catfile=options.input, hdu_index=options.hdu_index,
                                            rms=options.rms,
                                            outfile=options.outfile, bkgin=options.backgroundimg,
                                            rmsin=options.noiseimg, beam=options.beam, lat=lat, psf=options.psfimg,
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
                                           mask=options.region, lat=lat, psf=options.psfimg)
        if len(detections) == 0:
            log.info("No sources found in image")
        sources.extend(detections)

    if len(sources) > 0 and options.tables:
        for t in options.tables.split(','):
            save_catalog(t, sources)
