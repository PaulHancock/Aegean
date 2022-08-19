#! /usr/bin/env python
"""
The Aegean source finding program.
"""

import copy
import logging
import logging.config
import math
import multiprocessing
import os
import sys

import lmfit
import numpy as np
import scipy
from scipy.ndimage import find_objects, label, maximum_filter, minimum_filter
from scipy.special import erf
from tqdm import tqdm

from AegeanTools import wcs_helpers
from AegeanTools.fits_tools import write_fits

from . import cluster, flags
from .__init__ import __date__, __version__
from .angle_tools import bear, dec2dms, dec2hms, gcd
from .BANE import filter_image, get_step_size
from .catalogs import load_table, table_to_source_list
from .exceptions import AegeanNaNModelError
from .fits_tools import load_image_band
from .fitting import (Bmatrix, Cmatrix, bias_correct, covar_errors, do_lmfit,
                      elliptical_gaussian, errors, ntwodgaussian_lmfit)
from .models import (ComponentSource, DummyLM, GlobalFittingData,
                     IslandFittingData, IslandSource, PixelIsland,
                     SimpleSource, island_itergen)
from .msq2 import MarchingSquares
from .regions import Region
from .wcs_helpers import Beam, WCSHelper

__author__ = "Paul Hancock"

header = """#Aegean version {0}
# on dataset: {1}"""

# constants
CC2FHWM = 2 * math.sqrt(2 * math.log(2))
FWHM2CC = 1 / CC2FHWM

# dummy logger
log = logging.getLogger("dummy")
log.addHandler(logging.NullHandler())


def find_islands(im, bkg, rms,
                 seed_clip=5.0, flood_clip=4.0,
                 region=None, wcs=None,
                 log=log):
    """
    This function designed to be run as a stand alone process

    Parameters
    ----------
    im, bkg, rms : :class:`numpy.ndarray`
      Image, background, and rms maps

    seed_clip, flood_clip : float
      The seed clip which is used to create islands, and flood clip which is
      used to grow islands. The units are in SNR.

    region : :class:`AegeanTools.regions.Region` or None
      Region over which to find islands. Islands must have at least 1 pixel
      overlap with the region in order to be returned.
      Default = None, no constraints. Region *requires* a wcs to be defined.

    wcs : :class:`AegeanTools.wcs_helpers.WCSHelper` or None
      If a region is specified then this WCSHelper will be used to map sky->pix
      coordinates. Default = None.

    log : `logging.Logger` or None
      For handling logs (or not)

    Returns
    -------
    islands : [:class:`AegeanTools.models.PixelIsland`, ...]
      a list of islands
    """
    if (region is not None) and (wcs is None):
        log.warning("Find islands: Region was passed, but no wcs is defined." +
                    " Ignoring region.")
        region = None

    # compute SNR image
    snr = abs(im - bkg) / rms

    # mask of pixles that are above the flood_clip
    a = snr >= flood_clip

    if not np.any(a):
        log.debug("There are no pixels above the clipping limit")
        return []

    # segmentation via scipy
    # structure includes diagonal pixels as single island
    l, n = label(a, structure=np.ones((3, 3)))
    f = find_objects(l)

    log.debug("{1} Found {0} islands total above flood limit"
              .format(n, im.shape))

    islands = []
    for i in range(n):
        xmin, xmax = f[i][0].start, f[i][0].stop
        ymin, ymax = f[i][1].start, f[i][1].stop
        # obey seed clip constraint
        if np.any(snr[xmin:xmax, ymin:ymax] > seed_clip):
            # obey region constraint
            if region is not None:
                y, x = np.where(snr[xmin:xmax, ymin:ymax] >= flood_clip)
                yx = list(zip(y + ymin, x + xmin))
                ra, dec = wcs.wcs.wcs_pix2world(yx, 1).transpose()
                mask = region.sky_within(ra, dec, degin=True)
                if not np.any(mask):
                    continue

            # copy so that we don't blank the master data
            data_box = copy.deepcopy(im[xmin:xmax, ymin:ymax])

            # make mask and blank out pixels with below the noise level or
            # are pixels that are of another island in the FoV
            island_mask = (snr[xmin:xmax, ymin:ymax] < flood_clip) | \
                          (l[xmin:xmax, ymin:ymax] != i + 1)
            data_box[island_mask] = np.nan

            # check if there are any pixels left unmasked
            if not np.any(np.isfinite(data_box)):
                # self.log.info("{1} Island {0} has no non-masked pixels"
                #               .format(i,data.shape))
                continue

            island = PixelIsland()
            island.calc_bounding_box(
                np.array(np.nan_to_num(data_box), dtype=bool),
                offsets=[xmin, ymin]
            )
            island.set_mask(island_mask)
            islands.append(island)

    return islands


def estimate_parinfo_image(islands, im, rms, wcshelper,
                           max_summits=None, log=log):
    """
    Estimate the initial parameters for fitting for each of the islands of
    pixels. The source sizes will be initialised as the psf of the image,
    which is either determined by the WCS of the image file or the psf map
    if one is supplied.

    Parameters
    ----------
    islands : [AegeanTools.models.IslandFittingData, ... ]
      A list of islands which will be converted into groups of sources

    im, rms : :py:class:`numpy.ndarray`
      The image and noise maps

    wcshelper : :py:class:`AegeanTools.wcs_helpers.WCSHelper`
      A wcshelper object valid for the image map

    max_summits : int or None
      The maximum number of summits that will be fit. Any in addition to this
      will be estimated but their parameters will have vary=False.

    log : `logging.Logger` or None
      For handling logs (or not)

    max_summits : int
      The maximum number of summits that will be fit. Any in addition to this
      will be estimated but their parameters will have vary=False.

    log : `logging.Logger` or None
      For handling logs (or not)

    Returns
    --------
    sources : [:py:class:`lmfit.Parameters`, ... ]
      The initial estimate of parameters for the components within each island.
    """
    debug_on = log.isEnabledFor(logging.DEBUG)
    sources = []

    for island in islands:
        # set flags to be empty
        is_flag = 0x0
        [rmin, rmax], [cmin, cmax] = island.bounding_box
        i_data = copy.deepcopy(im[rmin:rmax, cmin:cmax])
        i_rms = copy.deepcopy(rms[rmin:rmax, cmin:cmax])

        # Mask out the bad pixels
        island_mask = island.mask
        i_data[island_mask] = np.nan
        i_rms[island_mask] = np.nan

        # the curvature needs a buffer of 1 pixel to correctly identify
        # the local min/max on the edge of the region.
        # We need a 1 pix buffer (if available)
        buffx = [rmin - max(rmin - 1, 0), min(rmax + 1, im.shape[0]) - rmax]
        buffy = [cmin - max(cmin - 1, 0), min(cmax + 1, im.shape[1]) - cmax]
        i_curve = np.zeros(
            shape=(
                rmax - rmin + buffx[0] + buffx[1],
                cmax - cmin + buffy[0] + buffy[1],
            ),
            dtype=np.int8,
        )
        # compute peaks and convert to +/-1
        peaks = maximum_filter(
            im[rmin - buffx[0]: rmax + buffx[1],
               cmin - buffy[0]: cmax + buffy[0]],
            size=3,
        )
        pmask = np.where(
            peaks == im[rmin-buffx[0]: rmax+buffx[1],
                        cmin-buffy[0]: cmax+buffy[0]]
        )
        troughs = minimum_filter(
            im[rmin - buffx[0]: rmax + buffx[1],
               cmin - buffy[0]: cmax + buffy[0]],
            size=3,
        )
        tmask = np.where(
            troughs == im[rmin-buffx[0]: rmax+buffx[1],
                          cmin-buffy[0]: cmax+buffy[0]]
        )
        i_curve[pmask] = -1
        i_curve[tmask] = 1

        # i_curve and im need to be the same size
        # so we crop i_curve based on the buffers that we computed
        i_curve = i_curve[
            buffx[0]: i_curve.shape[0] - buffx[1],
            buffy[0]: i_curve.shape[1] - buffy[1],
        ]
        del peaks, pmask, troughs, tmask, buffx, buffy

        # have already applied the island mask at start of loop
        isnegative = np.nanmax(i_data[np.isfinite(i_data)]) < 0

        # For small islands we can't do a 6 param fit
        # Don't count the NaN values as part of the island
        non_nan_pix = np.sum(np.isfinite(i_data))
        if 4 <= non_nan_pix <= 6:
            log.debug("FIXED2PSF")
            is_flag |= flags.FIXED2PSF
        elif non_nan_pix < 4:
            log.debug("FITERRSMALL!")
            is_flag |= flags.FITERRSMALL
        else:
            is_flag = 0
        if debug_on:
            log.debug(" - size {0}".format(len(i_data.ravel())))

        if (
            min(i_data.shape) <= 2
            or (is_flag & flags.FITERRSMALL)
            or (is_flag & flags.FIXED2PSF)
        ):
            # 1d islands or small islands only get one source
            if debug_on:
                log.debug("Tiny summit detected")
                log.debug("{0}".format(i_data))
            # and are constrained to be point sources
            is_flag |= flags.FIXED2PSF
            summits = [[slice(0, i_data.shape[0]), slice(0, i_data.shape[1])]]
            n = 1
        else:
            if isnegative:
                # the summit should be able to include all pixels within
                #  the island not just those above innerclip
                kappa_sigma = np.where(
                    i_curve > 0.5, np.where(np.isfinite(
                        i_data), i_data, np.nan), np.nan
                )
            else:
                kappa_sigma = np.where(
                    i_curve < -0.5,
                    np.where(np.isfinite(i_data), i_data, np.nan),
                    np.nan,
                )

            # count the number of peaks and their locations
            l, n = label(kappa_sigma)
            summits = find_objects(l)

        if n < 1:
            log.debug("Island has no summits")
            continue

        params = lmfit.Parameters()
        summits_considered = 0
        summits_accepted = 0
        # TODO: figure out how to sort the components in flux order

        for i in range(n):
            # x/y min/max are indices of the summit within the island
            xmin, xmax = summits[i][0].start, summits[i][0].stop
            ymin, ymax = summits[i][1].start, summits[i][1].stop
            summits_considered += 1
            summit_flag = is_flag

            summit = i_data[xmin:xmax, ymin:ymax]

            if debug_on:
                log.debug(
                    "Summit({0}) - shape: {1} x:[{2}-{3}] y:[{4}-{5}]".format(
                        i, summit.shape, ymin, ymax, xmin, xmax
                    )
                )
            try:
                if isnegative:
                    xpeak, ypeak = np.unravel_index(
                        np.nanargmin(summit), summit.shape)
                else:
                    xpeak, ypeak = np.unravel_index(
                        np.nanargmax(summit), summit.shape)
                amp = summit[xpeak, ypeak]
            except ValueError as e:
                if "All-NaN" in e.message:
                    log.warning(
                        "Summit of nan's detected - this shouldn't happen")
                    continue
                else:
                    raise e

            if debug_on:
                log.debug(" - max is {0:f}".format(amp))
                log.debug(" - peak at {0},{1}".format(xpeak, ypeak))

            # xo/yo are the index of the peak within the island
            yo = ypeak + ymin
            xo = xpeak + xmin

            # allow amp to be 5% or 3 sigma higher
            # NOTE: the 5% should depend on the beam sampling
            if amp > 0:
                amp_min, amp_max = (
                    0.95 * min(3 * i_rms[xo, yo], amp),
                    amp * 1.05 + 3 * i_rms[xo, yo],
                )
            else:
                amp_max, amp_min = (
                    0.95 * max(-3 * i_rms[xo, yo], amp),
                    amp * 1.05 - 3 * i_rms[xo, yo],
                )

            if debug_on:
                log.debug("a_min {0}, a_max {1}".format(amp_min, amp_max))

            # TODO: double check the yo/xo that seem reversed
            a, b, pa = wcshelper.get_psf_pix2pix(yo + cmin, xo + rmin)
            if not (np.all(np.isfinite((a, b, pa)))):
                log.debug(" Summit has invalid WCS/Beam - Skipping.")
                continue
            pixbeam = Beam(a, b, pa)

            # set a square limit based on the size of the pixbeam
            xo_lim = 0.5 * np.hypot(pixbeam.a, pixbeam.b)
            yo_lim = xo_lim
            yo_min, yo_max = yo - yo_lim, yo + yo_lim
            xo_min, xo_max = xo - xo_lim, xo + xo_lim

            # the size of the island
            xsize = i_data.shape[0]
            ysize = i_data.shape[1]

            # initial shape is the psf
            sx = pixbeam.a * FWHM2CC
            sy = pixbeam.b * FWHM2CC

            # lmfit does silly things if we start with these
            # two parameters being equal
            sx = max(sx, sy * 1.01)

            # constraints are based on the shape of the island
            # sx,sy can become flipped so we set the min/max account for this
            sx_min, sx_max = (
                sy * 0.8,
                max((max(xsize, ysize)+1) * math.sqrt(2) * FWHM2CC,
                    sx * 1.1)
            )
            sy_min, sy_max = (
                sy * 0.8,
                max((max(xsize, ysize) + 1) * math.sqrt(2) * FWHM2CC, sx * 1.1)
            )

            theta = pixbeam.pa  # Degrees
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
                log.debug(
                    " - sx {0} {1} {2} | {3} {4}".format(
                        sx, sx_min, sx_max, sx_min * CC2FHWM, sx_max * CC2FHWM
                    )
                )
                log.debug(
                    " - sy {0} {1} {2} | {3} {4}".format(
                        sy, sy_min, sy_max, sy_min * CC2FHWM, sy_max * CC2FHWM
                    )
                )
                log.debug(" - theta {0} {1} {2}".format(theta, -180, 180))
                log.debug(" - flags {0}".format(flag))
                log.debug(" - fit?  {0}".format(not maxxed))

            # TODO: figure out how incorporate the circular constraint on sx/sy
            prefix = "c{0}_".format(i)
            params.add(
                prefix + "amp",
                value=amp,
                min=amp_min,
                max=amp_max,
                vary=not maxxed
            )
            params.add(
                prefix + "xo",
                value=xo,
                min=float(xo_min),
                max=float(xo_max),
                vary=not maxxed,
            )
            params.add(
                prefix + "yo",
                value=yo,
                min=float(yo_min),
                max=float(yo_max),
                vary=not maxxed,
            )

            if summit_flag & flags.FIXED2PSF > 0:
                psf_vary = False
            else:
                psf_vary = not maxxed
            params.add(prefix + "sx", value=sx, min=sx_min,
                       max=sx_max, vary=psf_vary)
            params.add(prefix + "sy", value=sy, min=sy_min,
                       max=sy_max, vary=psf_vary)
            params.add(prefix + "theta", value=theta, vary=psf_vary)
            params.add(prefix + "flags", value=summit_flag, vary=False)

            summits_accepted += 1

        if debug_on:
            log.debug("Estimated sources: {0}".format(summits_accepted))
        # remember how many components are fit.
        params.add("components", value=summits_accepted, vary=False)

        if params["components"].value < n:
            log.debug(
                "Considered {0} summits, accepted {1}".format(
                    summits_considered, summits_accepted
                )
            )

        sources.append(params)

    return sources


class SourceFinder(object):
    """
    The Aegean source finding algorithm

    Attributes
    ----------
    global_data : :class:`AegeanTools.models.GlobalFittingData`
      State holder for properties.

    sources : list
      List of sources that have been found/measured.

    log : logging.log
      Logger to use.
      Default = None
    """

    def __init__(self, log=log, **kwargs):
        self.global_data = GlobalFittingData()
        self.sources = []
        self.log = log  # Use a dummy logger (which never reports anything)

        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
            else:
                print("{0} supplied but ignored".format(k))
        return

    def _gen_flood_wrap(self, data, rmsimg, innerclip,
                        outerclip=None, domask=False):
        """
        Generator function.
        Segment an image into islands and return one island at a time.

        Needs to work for entire image, and also for components
        within an island.

        Parameters
        ----------
        data : 2d-array
          Image array.

        rmsimg : 2d-array
          Noise image.

        innerclip, outerclip :float
          Seed (inner) and flood (outer) clipping values.

        domask : bool
          If True then look for a region mask in globals,
          only return islands that are within the region.
          Default = False.

        Yields
        ------
        data_box : 2d-array
          A island of sources with subthreshold values masked.

        xmin, xmax, ymin, ymax : int
          The corners of the data_box within the initial data array.
        """

        if outerclip is None:
            outerclip = innerclip

        # compute SNR image (data has already been background subtracted)
        snr = abs(data) / rmsimg
        # mask of pixles that are above the outerclip
        a = snr >= outerclip
        # segmentation a la scipy
        l, n = label(a)
        f = find_objects(l)

        if n == 0:
            self.log.debug("There are no pixels above the clipping limit")
            return
        self.log.debug(
            "{1} Found {0} islands total above flood limit".format(
                n, data.shape)
        )
        # Yield values as before, though they are not sorted by flux
        for i in range(n):
            xmin, xmax = f[i][0].start, f[i][0].stop
            ymin, ymax = f[i][1].start, f[i][1].stop
            if np.any(
                snr[xmin:xmax, ymin:ymax] > innerclip
            ):  # obey inner clip constraint
                # self.log.info("{1} Island {0} is above the inner clip limit"
                #               .format(i, data.shape))

                # Flag pixel that are either below the flood level
                # or belong to other islands that happen to be within
                # the bounding box
                island_mask = (snr[xmin:xmax, ymin:ymax] < outerclip) | \
                              (l[xmin:xmax, ymin:ymax] != i + 1)
                data_box = copy.deepcopy(
                    data[xmin:xmax, ymin:ymax]
                )  # copy so that we don't blank the master data

                # blank out the bad pixels
                data_box[island_mask] = np.nan
                # check if there are any pixels left unmasked
                if not np.any(np.isfinite(data_box)):
                    # self.log.info("{1} Island {0} has no non-masked pixels"
                    #               .format(i,data.shape))
                    continue

                if domask and (self.global_data.region is not None):
                    y, x = np.where(snr[xmin:xmax, ymin:ymax] >= outerclip)
                    # convert indices of this sub region to indices in
                    # the greater image
                    yx = list(zip(y + ymin, x + xmin))
                    ra, dec = self.global_data.wcshelper.wcs.wcs_pix2world(
                        yx, 1
                    ).transpose()
                    mask = self.global_data.region.sky_within(
                        ra, dec, degin=True)
                    # if there are no un-masked pixels within the region
                    # then we skip this island.
                    if not np.any(mask):
                        continue
                    self.log.debug("Mask {0}".format(mask))
                # self.log.info("{1} Island {0} will be fit"
                #               .format(i, data.shape))
                yield data_box, xmin, xmax, ymin, ymax

    ##
    # Estimating parameters, converting params -> sources,
    # and sources -> params
    ##
    def estimate_lmfit_parinfo(
        self,
        data,
        rmsimg,
        curve,
        beam,
        innerclip,
        outerclip=None,
        offsets=(0, 0),
        max_summits=None,
    ):
        """
        Estimates the number of sources in an island and returns
        initial parameters for the fit as well as limits on those parameters.

        Parameters
        ----------
        data : 2d-array
          (sub) image of flux values. Background should be subtracted.

        rmsimg : 2d-array
          Image of 1sigma values

        curve : 2d-array
          Image of curvature values [-1,0,+1]

        beam : :class:`AegeanTools.fits_image.Beam`
          The beam information for the image.

        innerclip, outerclip : float
          Inerr and outer level for clipping (sigmas).

        offsets : (int, int)
          The (x,y) offset of data within it's parent image

        max_summits : int
          If not None, only this many summits/components will be fit. More
          components may be present in the island, but subsequent components
          will not have free parameters.

        Returns
        -------
        model : lmfit.Parameters
          The initial estimate of parameters for the components
          within this island.
        """

        debug_on = self.log.isEnabledFor(logging.DEBUG)
        is_flag = 0
        global_data = self.global_data

        # check to see if this island is a negative peak since we need to
        # treat such cases slightly differently
        isnegative = np.nanmax(data[np.isfinite(data)]) < 0
        if isnegative:
            self.log.debug("[is a negative island]")

        if outerclip is None:
            outerclip = innerclip

        self.log.debug(" - shape {0}".format(data.shape))

        if not data.shape == curve.shape:
            self.log.error("data and curvature are mismatched")
            self.log.error("data:{0} curve:{1}".format(
                data.shape, curve.shape))
            raise AssertionError()

        # For small islands we can't do a 6 param fit
        # Don't count the NaN values as part of the island
        non_nan_pix = len(data[np.where(np.isfinite(data))].ravel())
        if 4 <= non_nan_pix <= 6:
            self.log.debug("FIXED2PSF")
            is_flag |= flags.FIXED2PSF
        elif non_nan_pix < 4:
            self.log.debug("FITERRSMALL!")
            is_flag |= flags.FITERRSMALL
        else:
            is_flag = 0
        if debug_on:
            self.log.debug(" - size {0}".format(len(data.ravel())))

        if (
            min(data.shape) <= 2
            or (is_flag & flags.FITERRSMALL)
            or (is_flag & flags.FIXED2PSF)
        ):
            # 1d islands or small islands only get one source
            if debug_on:
                self.log.debug("Tiny summit detected")
                self.log.debug("{0}".format(data))
            summits = [[data, 0, data.shape[0], 0, data.shape[1]]]
            # and are constrained to be point sources
            is_flag |= flags.FIXED2PSF
        else:
            if isnegative:
                # the summit should be able to include all pixels within
                # the island not just those above innerclip
                kappa_sigma = np.where(
                    curve > 0.5,
                    np.where(data + outerclip * rmsimg < 0, data, np.nan),
                    np.nan,
                )
            else:
                kappa_sigma = np.where(
                    -1 * curve > 0.5,
                    np.where(data - outerclip * rmsimg > 0, data, np.nan),
                    np.nan,
                )
            summits = list(
                self._gen_flood_wrap(
                    kappa_sigma, np.ones(kappa_sigma.shape), 0, domask=False
                )
            )

        params = lmfit.Parameters()
        i = 0
        summits_considered = 0
        # This can happen when the image contains regions of nans
        # the data/noise indicate an island,
        # but the curvature doesn't back it up.
        if len(summits) < 1:
            self.log.debug("Island has {0} summits".format(len(summits)))
            return None

        # add summits in reverse order of peak SNR - ie brightest first
        for summit, xmin, xmax, ymin, ymax in sorted(
            summits, key=lambda x: np.nanmax(-1.0 * abs(x[0]))
        ):
            summits_considered += 1
            summit_flag = is_flag
            if debug_on:
                self.log.debug(
                    "Summit({5}) - shape:{0} x:[{1}-{2}] y:[{3}-{4}]".format(
                        summit.shape, ymin, ymax, xmin, xmax, i
                    )
                )
            try:
                if isnegative:
                    amp = np.nanmin(summit)
                    xpeak, ypeak = np.unravel_index(
                        np.nanargmin(summit), summit.shape)
                else:
                    amp = np.nanmax(summit)
                    xpeak, ypeak = np.unravel_index(
                        np.nanargmax(summit), summit.shape)
            except ValueError as e:
                if "All-NaN" in e.message:
                    self.log.warning(
                        "Summit of nan's detected - this shouldn't happen")
                    continue
                else:
                    raise e

            if debug_on:
                self.log.debug(" - max is {0:f}".format(amp))
                self.log.debug(" - peak at {0},{1}".format(xpeak, ypeak))
            yo = ypeak + ymin
            xo = xpeak + xmin

            # Summits are allowed to include pixels that are between the
            # outer and inner clip. This means that sometimes we get
            # a summit that has all it's pixels below the inner clip.
            # So we test for that here.
            snr = np.nanmax(
                abs(
                    data[xmin: xmax + 1, ymin: ymax + 1]
                    / rmsimg[xmin: xmax + 1, ymin: ymax + 1]
                )
            )
            if snr < innerclip:
                self.log.debug(
                    "Summit has SNR {0} < innerclip {1}: skipping".format(
                        snr, innerclip
                    )
                )
                continue

            # allow amp to be 5% or (innerclip) sigma higher
            # TODO: the 5% should depend on the beam sampling
            # note: when innerclip is 400 this becomes rather stupid
            if amp > 0:
                amp_min, amp_max = (
                    0.95 * min(outerclip * rmsimg[xo, yo], amp),
                    amp * 1.05 + innerclip * rmsimg[xo, yo],
                )
            else:
                amp_max, amp_min = (
                    0.95 * max(-outerclip * rmsimg[xo, yo], amp),
                    amp * 1.05 - innerclip * rmsimg[xo, yo],
                )

            if debug_on:
                self.log.debug("a_min {0}, a_max {1}".format(amp_min, amp_max))

            a, b, pa = global_data.psfhelper.get_psf_pix2pix(
                yo + offsets[0], xo + offsets[1]
            )
            if not (np.all(np.isfinite((a, b, pa)))):
                self.log.debug(" Summit has invalid WCS/Beam - Skipping.")
                continue
            pixbeam = Beam(a, b, pa)

            # set a square limit based on the size of the pixbeam
            xo_lim = 0.5 * np.hypot(pixbeam.a, pixbeam.b)
            yo_lim = xo_lim

            yo_min, yo_max = yo - yo_lim, yo + yo_lim

            xo_min, xo_max = xo - xo_lim, xo + xo_lim

            # the size of the island
            xsize = data.shape[0]
            ysize = data.shape[1]

            # initial shape is the psf
            sx = pixbeam.a * FWHM2CC
            sy = pixbeam.b * FWHM2CC

            # lmfit does silly things if we start with these
            # two parameters being equal
            sx = max(sx, sy * 1.01)

            # constraints are based on the shape of the island
            # sx,sy can become flipped so we set the min/max account for this
            sx_min, sx_max = (
                sy * 0.8,
                max((max(xsize, ysize) + 1) * math.sqrt(2) * FWHM2CC,
                    sx * 1.1)
            )
            sy_min, sy_max = (
                sy * 0.8,
                max((max(xsize, ysize) + 1) * math.sqrt(2) * FWHM2CC,
                    sx * 1.1)
            )

            theta = pixbeam.pa  # Degrees
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
                self.log.debug(" - var val min max | min max")
                self.log.debug(
                    " - amp {0} {1} {2} ".format(amp, amp_min, amp_max))
                self.log.debug(" - xo {0} {1} {2} ".format(xo, xo_min, xo_max))
                self.log.debug(" - yo {0} {1} {2} ".format(yo, yo_min, yo_max))
                self.log.debug(
                    " - sx {0} {1} {2} | {3} {4}".format(
                        sx, sx_min, sx_max, sx_min * CC2FHWM, sx_max * CC2FHWM
                    )
                )
                self.log.debug(
                    " - sy {0} {1} {2} | {3} {4}".format(
                        sy, sy_min, sy_max, sy_min * CC2FHWM, sy_max * CC2FHWM
                    )
                )
                self.log.debug(" - theta {0} {1} {2}".format(theta, -180, 180))
                self.log.debug(" - flags {0}".format(flag))
                self.log.debug(" - fit?  {0}".format(not maxxed))

            # TODO: figure out how incorporate the circular constraint on sx/sy
            prefix = "c{0}_".format(i)
            params.add(
                prefix + "amp",
                value=amp,
                min=amp_min,
                max=amp_max,
                vary=not maxxed
            )
            params.add(
                prefix + "xo",
                value=xo,
                min=float(xo_min),
                max=float(xo_max),
                vary=not maxxed,
            )
            params.add(
                prefix + "yo",
                value=yo,
                min=float(yo_min),
                max=float(yo_max),
                vary=not maxxed,
            )

            if summit_flag & flags.FIXED2PSF > 0:
                psf_vary = False
            else:
                psf_vary = not maxxed
            params.add(prefix + "sx", value=sx, min=sx_min,
                       max=sx_max, vary=psf_vary)
            params.add(prefix + "sy", value=sy, min=sy_min,
                       max=sy_max, vary=psf_vary)
            params.add(prefix + "theta", value=theta, vary=psf_vary)
            params.add(prefix + "flags", value=summit_flag, vary=False)

            # starting at zero allows the maj/min axes to be fit better.
            # if params[prefix + 'theta'].vary:
            #     params[prefix + 'theta'].value = 0

            i += 1
        if debug_on:
            self.log.debug("Estimated sources: {0}".format(i))
        # remember how many components are fit.
        params.add("components", value=i, vary=False)
        # params.components=i
        if params["components"].value < 1:
            self.log.debug(
                "Considered {0} summits, accepted {1}".format(
                    summits_considered, i)
            )
        return params

    def result_to_components(self, result, model, island_data, isflags):
        """
        Convert fitting results into a set of components

        Parameters
        ----------
        result : lmfit.MinimizerResult
          The fitting results.

        model : lmfit.Parameters
          The model that was fit.

        island_data : :class:`AegeanTools.models.IslandFittingData`
          Data about the island that was fit.

        isflags : int
          Flags that should be added to this island
          in addition to those within the model)

        Returns
        -------
        sources : list
          A list of components, and islands if requested.
        """
        global_data = self.global_data

        # island data
        isle_num = island_data.isle_num
        idata = island_data.i
        xmin, xmax, ymin, ymax = island_data.offsets

        box = slice(int(xmin), int(xmax)), slice(int(ymin), int(ymax))
        rms = global_data.rmsimg[box]
        bkg = global_data.bkgimg[box]
        residual = np.median(result.residual), np.std(result.residual)
        is_flag = isflags

        sources = []
        j = 0
        for j in range(model["components"].value):
            src_flags = is_flag
            source = ComponentSource()
            source.island = isle_num
            source.source = j
            self.log.debug(" component {0}".format(j))
            prefix = "c{0}_".format(j)
            xo = model[prefix + "xo"].value
            yo = model[prefix + "yo"].value
            sx = model[prefix + "sx"].value
            sy = model[prefix + "sy"].value
            theta = model[prefix + "theta"].value
            amp = model[prefix + "amp"].value
            src_flags |= model[prefix + "flags"].value

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
            # update the source xo/yo so the error calculations can be done
            # correctly.  Note that you have to update the max or the value
            # you set will be clipped at the max allowed value
            model[prefix + "xo"].set(value=x_pix, max=np.inf)
            model[prefix + "yo"].set(value=y_pix, max=np.inf)
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
            (
                source.ra,
                source.dec,
                source.a,
                source.b,
                source.pa,
            ) = global_data.wcshelper.pix2sky_ellipse(
                (x_pix, y_pix), sx * CC2FHWM, sy * CC2FHWM, theta
            )
            source.a *= 3600  # arcseconds
            source.b *= 3600
            # force a>=b
            fix_shape(source)
            # limit the pa to be in (-90,90]
            source.pa = pa_limit(source.pa)

            # if one of these values are nan then there has been some problem
            # with the WCS handling
            if not all(
                np.isfinite(
                    (source.ra, source.dec, source.a, source.b, source.pa))
            ):
                src_flags |= flags.WCSERR
            # negative degrees is valid for RA, but I don't want them.
            if source.ra < 0:
                source.ra += 360
            source.ra_str = dec2hms(source.ra)
            source.dec_str = dec2dms(source.dec)

            # calculate integrated flux
            source.int_flux = source.peak_flux * sx * sy * CC2FHWM ** 2 * np.pi
            # scale Jy/beam -> Jy using the area of the beam
            source.int_flux /= global_data.psfhelper.get_beamarea_pix(
                source.ra, source.dec
            )

            # Calculate errors for params that were fit (as well as int_flux)
            errors(source, model, global_data.wcshelper)

            source.flags = src_flags
            # add psf info
            local_beam = global_data.psfhelper.get_skybeam(
                source.ra, source.dec)
            if local_beam is not None:
                source.psf_a = local_beam.a * 3600
                source.psf_b = local_beam.b * 3600
                source.psf_pa = local_beam.pa
            else:
                source.psf_a = 0
                source.psf_b = 0
                source.psf_pa = 0
            sources.append(source)
            self.log.debug(source)

        if global_data.blank:
            outerclip = island_data.scalars[1]
            idx, idy = np.where(abs(idata) - outerclip * rms > 0)
            idx += xmin
            idy += ymin
            self.global_data.img[[idx, idy]] = np.nan

        # calculate the integrated island flux if required
        if island_data.doislandflux:
            _, outerclip, _ = island_data.scalars
            self.log.debug("Integrated flux for island {0}".format(isle_num))
            kappa_sigma = np.where(
                abs(idata) - outerclip * rms > 0, idata, np.NaN)
            self.log.debug("- island shape is {0}".format(kappa_sigma.shape))

            source = IslandSource()
            source.flags = 0
            source.island = isle_num
            source.components = j + 1
            source.peak_flux = np.nanmax(kappa_sigma)
            # check for negative islands
            if source.peak_flux < 0:
                source.peak_flux = np.nanmin(kappa_sigma)
            self.log.debug("- peak flux {0}".format(source.peak_flux))

            # positions and background
            # if a component has been refit then it might have flux = np.nan
            if np.isfinite(source.peak_flux):
                positions = np.where(kappa_sigma == source.peak_flux)
            else:
                positions = [[kappa_sigma.shape[0] / 2],
                             [kappa_sigma.shape[1] / 2]]
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

            # TODO: investigate what happens when the sky coords are
            #       skewed w.r.t the pixel coords

            # calculate the area of the island as a fraction of the
            # area of the bounding box
            bl = global_data.wcshelper.pix2sky([xmax, ymin])
            tl = global_data.wcshelper.pix2sky([xmax, ymax])
            tr = global_data.wcshelper.pix2sky([xmin, ymax])
            height = gcd(tl[0], tl[1], bl[0], bl[1])
            width = gcd(tl[0], tl[1], tr[0], tr[1])
            area = height * width
            source.area = (
                area * source.pixels / source.x_width / source.y_width
            )  # area is in deg^2

            # create contours around the data which was used in fitting
            msq = MarchingSquares(kappa_sigma)
            source.contour = [(a[0] + xmin, a[1] + ymin)
                              for a in msq.perimeter]
            # calculate the maximum angular size of this island,
            # using a brute force method
            source.max_angular_size = 0
            for i, pos1 in enumerate(source.contour):
                radec1 = global_data.wcshelper.pix2sky(pos1)
                for j, pos2 in enumerate(source.contour[i:]):
                    radec2 = global_data.wcshelper.pix2sky(pos2)
                    dist = gcd(radec1[0], radec1[1], radec2[0], radec2[1])
                    if dist > source.max_angular_size:
                        source.max_angular_size = dist
                        source.pa = bear(
                            radec1[0], radec1[1], radec2[0], radec2[1])
                        source.max_angular_size_anchors = [
                            pos1[0],
                            pos1[1],
                            pos2[0],
                            pos2[1],
                        ]

            self.log.debug(
                "- peak position {0}, {1} [{2},{3}]"
                .format(source.ra_str, source.dec_str,
                        positions[0][0], positions[1][0])
            )

            # integrated flux
            beam_area_pix = global_data.psfhelper.get_beamarea_pix(
                source.ra, source.dec
            )
            beam_area = global_data.psfhelper.get_beamarea_deg2(
                source.ra, source.dec)
            isize = source.pixels  # number of non zero pixels
            self.log.debug("- pixels used {0}".format(isize))
            source.int_flux = np.nansum(kappa_sigma)  # total flux Jy/beam
            self.log.debug("- sum of pixles {0}".format(source.int_flux))
            source.int_flux *= 4.0 * \
                np.log(2.0) / beam_area_pix  # total flux in Jy
            self.log.debug("- integrated flux {0}".format(source.int_flux))
            eta = erf(np.sqrt(-1*np.log(abs(source.local_rms*outerclip
                                            / source.peak_flux))))**2
            self.log.debug("- eta {0}".format(eta))
            source.eta = eta
            source.beam_area = beam_area

            # I don't know how to calculate this error so we'll set it to nan
            source.err_int_flux = np.nan
            sources.append(source)
        return sources

    ##
    # Setting up 'global' data and calculating bkg/rms
    ##
    def load_globals(
        self,
        filename,
        hdu_index=0,
        bkgin=None,
        rmsin=None,
        beam=None,
        verb=False,
        rms=None,
        bkg=None,
        cores=1,
        do_curve=False,
        mask=None,
        psf=None,
        blank=False,
        docov=True,
        cube_index=None,
    ):
        """
        Populate the global_data object by loading or calculating
        the various components

        Parameters
        ----------
        filename : str or HDUList
          Main image which source finding is run on

        hdu_index : int
          HDU index of the image within the fits file, default is 0 (first)

        bkgin, rmsin : str or HDUList
          background and noise image filename or HDUList

        beam : :class:`AegeanTools.fits_image.Beam`
          Beam object representing the synthsized beam.
          Will replace what is in the FITS header.

        verb : bool
          Verbose. Write extra lines to INFO level log.

        rms, bkg : float
          A float that represents a constant rms/bkg levels for the image.
          Default = None, which causes the rms/bkg to be loaded or calculated.

        cores : int
          Number of cores to use if different from what is autodetected.

        do_curve : bool
          If True a curvature map will be created, default=True.

        mask : str or :class:`AegeanTools.regions.Region`
          filename or Region object

        psf : str or HDUList
          Filename or HDUList of a psf image

        blank : bool
          True = blank output image where islands are found.
          Default = False.

        docov : bool
          True = use covariance matrix in fitting.
          Default = True.

        cube_index : int
          For an image cube, which slice to use.

        """
        # don't reload already loaded data
        if self.global_data.img is not None:
            return
        # img = FitsImage(filename, hdu_index=hdu_index,
        #                 beam=beam, cube_index=cube_index)
        img, header = load_image_band(filename,
                                      hdu_index=hdu_index,
                                      cube_index=cube_index)

        debug = logging.getLogger("Aegean").isEnabledFor(logging.DEBUG)

        if mask is None:
            self.global_data.region = None
        else:
            # allow users to supply and object instead of a filename
            if isinstance(mask, Region):
                self.global_data.region = mask
            elif os.path.exists(mask):
                self.log.info("Loading mask from {0}".format(mask))
                self.global_data.region = Region.load(mask)
            else:
                self.log.error("File {0} not found for loading".format(mask))
                self.global_data.region = None

        self.global_data.wcshelper = WCSHelper.from_header(
            header, beam, psf_file=psf
        )
        self.global_data.psfhelper = self.global_data.wcshelper

        self.global_data.beam = self.global_data.wcshelper.beam
        self.global_data.img = img
        self.global_data.header = header
        self.global_data.dtype = type(self.global_data.img[0][0])
        self.global_data.bkgimg = np.zeros(
            self.global_data.img.shape, dtype=self.global_data.dtype
        )
        self.global_data.rmsimg = np.zeros(
            self.global_data.img.shape, dtype=self.global_data.dtype
        )
        self.global_data.pixarea = wcs_helpers.get_pixinfo(header)[0]
        self.global_data.dcurve = None
        self.global_data.cube_index = cube_index

        if do_curve:
            self.log.info("Calculating curvature")
            # calculate curvature but store it as -1,0,+1
            dcurve = np.zeros(self.global_data.img.shape, dtype=np.int8)
            peaks = maximum_filter(self.global_data.img, size=3)
            troughs = minimum_filter(self.global_data.img, size=3)
            pmask = np.where(self.global_data.img == peaks)
            tmask = np.where(self.global_data.img == troughs)
            dcurve[pmask] = -1
            dcurve[tmask] = 1
            self.global_data.dcurve = dcurve

        # if either of rms or bkg images are not supplied
        # then calculate them both
        if not (rmsin and bkgin):

            if verb:
                self.log.info("Calculating background and rms data")
            self._make_bkg_rms(
                filename=filename,
                forced_rms=rms,
                forced_bkg=bkg,
                cores=cores,
            )

        # replace the calculated images with input versions,
        # if the user has supplied them.
        if bkgin:
            if verb:
                self.log.info(
                    "Loading background data from file {0}".format(bkgin))
            self.global_data.bkgimg = self._load_aux_image(img, bkgin)
        if rmsin:
            if verb:
                self.log.info("Loading rms data from file {0}".format(rmsin))
            self.global_data.rmsimg = self._load_aux_image(img, rmsin)

        # subtract the background image from the data image and save
        if verb and debug:
            self.log.debug(
                "Data max is {0}".format(
                    np.nanmax(img)
                )
            )
            self.log.debug("Doing background subtraction")
        img -= self.global_data.bkgimg
        self.global_data.img = img
        if verb and debug:
            self.log.debug(
                "Data max is {0}".format(
                    np.nanmax(img)
                )
            )

        self.global_data.blank = blank
        self.global_data.docov = docov

        # Default to false until I can verify that this is working
        self.global_data.dobias = False

        # check if the WCS is galactic
        if "lon" in self.global_data.header["CTYPE1"].lower():
            self.log.info("Galactic coordinates detected and noted")
            SimpleSource.galactic = True
        return

    def save_background_files(
        self,
        image_filename,
        hdu_index=0,
        bkgin=None,
        rmsin=None,
        beam=None,
        rms=None,
        bkg=None,
        cores=1,
        outbase=None,
        cube_index=None,
    ):
        """
        Generate and save the background and RMS maps as FITS files.
        They are saved in the current directly as aegean-background.fits
        and aegean-rms.fits.

        Parameters
        ----------
        image_filename : str or HDUList
          Input image.

        hdu_index : int
          If fits file has more than one hdu, it can be specified here.
          Default = 0.

        bkgin, rmsin : str or HDUList
          Background and noise image filename or HDUList

        beam : :class:`AegeanTools.fits_image.Beam`
          Beam object representing the synthsized beam.
          Will replace what is in the FITS header.

        rms, bkg : float
          A float that represents a constant rms/bkg level for the image.
          Default = None, which causes the rms/bkg to be loaded or calculated.

        cores : int
          Number of cores to use if different from what is autodetected.

        outbase : str
          Basename for output files.

        cube_index : int
          If images are 3d, use this index into the 3rd axis.
        """

        self.log.info("Saving background / RMS maps")
        # load image, and load/create background/rms images
        self.load_globals(
            image_filename,
            hdu_index=hdu_index,
            bkgin=bkgin,
            rmsin=rmsin,
            beam=beam,
            verb=True,
            rms=rms,
            bkg=bkg,
            cores=cores,
            do_curve=True,
            cube_index=cube_index
        )
        img = self.global_data.img
        bkgimg, rmsimg = self.global_data.bkgimg, self.global_data.rmsimg
        curve = np.array(self.global_data.dcurve, dtype=bkgimg.dtype)
        # mask these arrays have the same mask the same as the data
        mask = np.where(np.isnan(img))
        bkgimg[mask] = np.NaN
        rmsimg[mask] = np.NaN
        curve[mask] = np.NaN

        # Generate the new FITS files by copying the existing HDU
        # and assigning new data. This gives the new files the same
        # WCS projection and other header fields.
        header = self.global_data.header
        # Set the ORIGIN to indicate Aegean made this file
        header["ORIGIN"] = "Aegean {0}-({1})".format(
            __version__, __date__)
        for c in [
            "CRPIX3",
            "CRPIX4",
            "CDELT3",
            "CDELT4",
            "CRVAL3",
            "CRVAL4",
            "CTYPE3",
            "CTYPE4",
        ]:
            if c in header:
                del header[c]

        if outbase is None:
            outbase, _ = os.path.splitext(os.path.basename(image_filename))
        noise_out = outbase + "_rms.fits"
        background_out = outbase + "_bkg.fits"
        curve_out = outbase + "_crv.fits"
        snr_out = outbase + "_snr.fits"

        write_fits(bkgimg, header, background_out)
        self.log.info("Wrote {0}".format(background_out))

        write_fits(rmsimg, header, noise_out)
        self.log.info("Wrote {0}".format(noise_out))

        write_fits(curve, header, curve_out)
        self.log.info("Wrote {0}".format(curve_out))

        write_fits(img/rmsimg, header, snr_out)
        self.log.info("Wrote {0}".format(snr_out))
        return

    def save_image(self, outname):
        """
        Save the image data.
        This is probably only useful if the image data has been blanked.

        Parameters
        ----------
        outname : str
          Name for the output file.
        """
        header = self.global_data.header
        header["ORIGIN"] = "Aegean {0}-({1})".format(__version__, __date__)
        # delete some axes that we aren't going to need
        for c in [
            "CRPIX3",
            "CRPIX4",
            "CDELT3",
            "CDELT4",
            "CRVAL3",
            "CRVAL4",
            "CTYPE3",
            "CTYPE4",
        ]:
            if c in header:
                del header[c]
        write_fits(self.global_data.img, header, outname)
        self.log.info("Wrote {0}".format(outname))
        return

    def _make_bkg_rms(self, filename,
                      forced_rms=None, forced_bkg=None,
                      cores=None):
        """
        Calculate an rms image and a bkg image.

        Parameters
        ----------
        filename : str
          Path of the image file.

        forced_rms : float
          The rms of the image.
          If None:  calculate the rms level (default).
          Otherwise assume a constant rms.

        forced_bkg : float
          The background level of the image.
          If None: calculate the background level (default).
          Otherwise assume a constant background.

        cores: int
          Number of cores to use if different from what is autodetected.

        """
        if forced_rms is not None:
            self.log.info("Forcing rms = {0}".format(forced_rms))
            self.global_data.rmsimg[:] = forced_rms
        if forced_bkg is not None:
            self.log.info("Forcing bkg = {0}".format(forced_bkg))
            self.global_data.bkgimg[:] = forced_bkg

        # If we known both the rms and the bkg then there is nothing to compute
        if (forced_rms is not None) and (forced_bkg is not None):
            return

        # use the BANE background/rms calculation
        step_size = get_step_size(self.global_data.header)
        box_size = (5 * step_size[0], 5 * step_size[1])

        bkg, rms = filter_image(
            im_name=filename,
            out_base=None,
            step_size=step_size,
            box_size=box_size,
            cores=cores,
            cube_index=self.global_data.cube_index,
        )
        if forced_rms is None:
            self.global_data.rmsimg = rms
        if forced_bkg is None:
            self.global_data.bkgimg = bkg

        return

    def _load_aux_image(self, image, auxfile):
        """
        Load a fits file (bkg/rms/curve) and make sure that
        it is the same shape as the main image.

        Parameters
        ----------
        image : :class:`numpy.ndarray`
          The main image that has already been loaded.

        auxfile : str
          The auxiliary file to be loaded.

        Returns
        -------
        aux : :class:`numpy.ndarray`
          The loaded image.
        """

        auximg, _ = load_image_band(auxfile)
        if auximg.shape != image.shape:
            self.log.error(
                "file {0} is not the same size as the image map".format(
                    auxfile)
            )
            self.log.error(
                "{0}= {1}, image = {2}".format(
                    auxfile, auximg.shape, image.shape
                )
            )
            sys.exit(1)
        return auximg

    ##
    # Fitting and refitting
    ##
    def _refit_islands(self, group, stage, outerclip=None, istart=0):
        """
        Do island refitting (priorized fitting) on a group of islands.

        Parameters
        ----------
        group : list
          A list of components grouped by island.

        stage : int
          Refitting stage.

        outerclip : float
          Ignored, placed holder for future development.

        istart : int
          The starting island number.

        Returns
        -------
        sources : list
          List of sources (and islands).
        """
        global_data = self.global_data
        sources = []

        data = global_data.img
        rmsimg = global_data.rmsimg

        for inum, isle in enumerate(group, start=istart):
            self.log.debug("-=-")
            self.log.debug(
                "input island = {0}, {1} components".format(
                    isle[0].island, len(isle))
            )

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
                pixbeam = Beam(
                    *global_data.psfhelper.get_psf_sky2pix(src.ra, src.dec))
                # find the right pixels from the ra/dec
                source_x, source_y = global_data.wcshelper.sky2pix(
                    [src.ra, src.dec])
                source_x -= 1
                source_y -= 1
                x = int(round(source_x))
                y = int(round(source_y))

                self.log.debug(
                    "pixel location ({0:5.2f},{1:5.2f})".format(
                        source_x, source_y)
                )
                # reject sources that are outside the image bounds,
                # or which have nan data/rms values
                if (
                    not 0 <= x < shape[0]
                    or not 0 <= y < shape[1]
                    or not np.isfinite(data[x, y])
                    or not np.isfinite(rmsimg[x, y])
                    or pixbeam is None
                ):
                    self.log.debug(
                        "Source ({0},{1}) not within usable region: skipping"
                        .format(src.island, src.source)
                    )
                    continue
                else:
                    # Keep track of the last source to have a valid psf
                    # so that we can use it later on
                    src_valid_psf = src
                # determine the shape parameters in pixel values
                _, _, sx, sy, theta = global_data.wcshelper.sky2pix_ellipse(
                    [src.ra, src.dec], src.a / 3600, src.b / 3600, src.pa
                )
                sx *= FWHM2CC
                sy *= FWHM2CC

                self.log.debug(
                    "Source shape [sky coords]  {0:5.2f}x{1:5.2f}@{2:05.2f}"
                    .format(src.a, src.b, src.pa)
                )
                self.log.debug(
                    "Source shape [pixel coords] {0:4.2f}x{1:4.2f}@{2:05.2f}"
                    .format(sx, sy, theta)
                )

                # choose a region that is 2x the major axis of the source,
                # 4x semimajor axis a
                width = 4 * sx
                ywidth = int(round(width)) + 1
                xwidth = int(round(width)) + 1

                # adjust the size of the island to include this source
                xmin = min(xmin, max(0, x - xwidth / 2))
                ymin = min(ymin, max(0, y - ywidth / 2))
                xmax = max(xmax, min(shape[0], x + xwidth / 2 + 1))
                ymax = max(ymax, min(shape[1], y + ywidth / 2 + 1))

                s_lims = [0.8 * min(sx, pixbeam.b * FWHM2CC),
                          max(sy, sx) * 1.25]

                # Set up the parameters for the fit, including constraints
                prefix = "c{0}_".format(i)
                params.add(prefix + "amp", value=src.peak_flux, vary=True)
                # for now the xo/yo are locations within the main image,
                # we correct this later
                params.add(
                    prefix + "xo",
                    value=source_x,
                    min=source_x - sx / 2.0,
                    max=source_x + sx / 2.0,
                    vary=stage >= 2,
                )
                params.add(
                    prefix + "yo",
                    value=source_y,
                    min=source_y - sy / 2.0,
                    max=source_y + sy / 2.0,
                    vary=stage >= 2,
                )
                params.add(
                    prefix + "sx",
                    value=sx,
                    min=s_lims[0],
                    max=s_lims[1],
                    vary=stage >= 3,
                )
                params.add(
                    prefix + "sy",
                    value=sy,
                    min=s_lims[0],
                    max=s_lims[1],
                    vary=stage >= 3,
                )
                params.add(prefix + "theta", value=theta, vary=stage >= 3)
                params.add(prefix + "flags", value=0, vary=False)
                # this source is being refit so add it to the list
                included_sources.append(src)
                i += 1

                # TODO: Allow this mask to be used in conjunction with
                # the FWHM mask that is defined further on

            if i == 0:
                self.log.debug(
                    "No sources found in island {0}".format(src.island))
                continue
            params.add("components", value=i, vary=False)
            # params.components = i
            self.log.debug(" {0} components being fit".format(i))
            # now we correct the xo/yo positions to be
            # relative to the sub-image
            self.log.debug("xmxxymyx {0} {1} {2} {3}".format(
                xmin, xmax, ymin, ymax))
            for i in range(params["components"].value):
                prefix = "c{0}_".format(i)
                # must update limits before the value as limits are
                # enforced when the value is updated
                params[prefix + "xo"].min -= xmin
                params[prefix + "xo"].max -= xmin
                params[prefix + "xo"].value -= xmin
                params[prefix + "yo"].min -= ymin
                params[prefix + "yo"].max -= ymin
                params[prefix + "yo"].value -= ymin
            # self.log.debug(params)
            # don't fit if there are no sources
            if params["components"].value < 1:
                self.log.info(
                    "Island {0} has no components".format(src.island))
                continue

            # this .copy() will stop us from modifying the parent region when
            # we later apply our mask.
            idata = data[int(xmin): int(xmax), int(ymin): int(ymax)].copy()
            # now convert these back to indices within the idata region
            # island_mask = np.array([(x-xmin, y-ymin) for x,y in island_mask])

            allx, ally = np.indices(idata.shape)
            # mask to include pixels that are withn the FWHM
            # of the sources being fit
            mask_params = copy.deepcopy(params)
            for i in range(mask_params["components"].value):
                prefix = "c{0}_".format(i)
                mask_params[prefix + "amp"].value = 1
            mask_model = ntwodgaussian_lmfit(mask_params)
            mask = np.where(mask_model(allx.ravel(), ally.ravel()) <= 0.1)
            mask = allx.ravel()[mask], ally.ravel()[mask]
            del mask_params

            idata[mask] = np.nan

            mx, my = np.where(np.isfinite(idata))
            non_nan_pix = len(mx)
            total_pix = len(allx.ravel())
            self.log.debug("island extracted:")
            self.log.debug(" x[{0}:{1}] y[{2}:{3}]".format(
                xmin, xmax, ymin, ymax))
            self.log.debug(" max = {0}".format(np.nanmax(idata)))
            self.log.debug(
                " total {0}, masked {1}, not masked {2}".format(
                    total_pix, total_pix - non_nan_pix, non_nan_pix
                )
            )

            # Check to see that each component has some data within
            # the central 3x3 pixels of it's location
            # If not then we don't fit that component
            for i in range(params["components"].value):
                prefix = "c{0}_".format(i)
                # figure out a box around the center of this
                cx, cy = (
                    params[prefix + "xo"].value,
                    params[prefix + "yo"].value,
                )  # central pixel coords
                self.log.debug(" comp {0}".format(i))
                self.log.debug("  x0, y0 {0} {1}".format(cx, cy))
                xmx = int(round(np.clip(cx + 2, 0, idata.shape[0])))
                xmn = int(round(np.clip(cx - 1, 0, idata.shape[0])))
                ymx = int(round(np.clip(cy + 2, 0, idata.shape[1])))
                ymn = int(round(np.clip(cy - 1, 0, idata.shape[1])))
                square = idata[xmn:xmx, ymn:ymx]
                # if there are no not-nan pixels in this region
                # then don't vary any parameters
                if not np.any(np.isfinite(square)):
                    self.log.debug(" not fitting component {0}".format(i))
                    params[prefix + "amp"].value = np.nan
                    for p in ["amp", "xo", "yo", "sx", "sy", "theta"]:
                        params[prefix + p].vary = False
                        params[prefix + p].stderr = np.nan
                        # the above results in an error of -1 later on
                    params[prefix + "flags"].value |= flags.NOTFIT

            # determine the number of free parameters and
            # if we have enough data for a fit
            nfree = np.count_nonzero([params[p].vary for p in params.keys()])
            self.log.debug(params)
            if nfree < 1:
                self.log.debug(" Island has no components to fit")
                result = DummyLM()
                model = params
            else:
                if non_nan_pix < nfree:
                    self.log.debug(
                        "More free parameters {0} than available pixels {1}"
                        .format(nfree, non_nan_pix)
                    )
                    if non_nan_pix >= params["components"].value:
                        self.log.debug(
                            "Fixing all parameters except amplitudes")
                        for p in params.keys():
                            if "amp" not in p:
                                params[p].vary = False
                    else:
                        self.log.debug(" no not-masked pixels, skipping")
                    continue

                # do the fit
                # if the pixel beam is not valid, then recalculate using
                # the location of the last source to have a valid psf
                if pixbeam is None:
                    if src_valid_psf is not None:
                        pixbeam = global_data.psfhelper.get_pixbeam(
                            src_valid_psf.ra, src_valid_psf.dec
                        )
                    else:
                        self.log.critical("Cannot determine pixel beam")
                fac = 1 / np.sqrt(2)
                if self.global_data.docov:
                    C = Cmatrix(
                        mx,
                        my,
                        pixbeam.a * FWHM2CC * fac,
                        pixbeam.b * FWHM2CC * fac,
                        pixbeam.pa,
                    )
                    B = Bmatrix(C)
                else:
                    C = B = None
                errs = np.nanmax(
                    rmsimg[int(xmin): int(xmax), int(ymin): int(ymax)])
                result, _ = do_lmfit(idata, params, B=B)
                model = covar_errors(result.params, idata, errs=errs, B=B, C=C)

            # convert the results to a source object
            offsets = (xmin, xmax, ymin, ymax)
            # TODO allow for island fluxes in the refitting.
            island_data = IslandFittingData(
                inum,
                i=idata,
                offsets=offsets,
                doislandflux=False,
                scalars=(4, 4, None)
            )
            new_src = self.result_to_components(
                result, model, island_data, src.flags)

            for ns, s in zip(new_src, included_sources):
                # preserve the uuid so we can do exact
                # matching between catalogs
                ns.uuid = s.uuid

                # flag the sources as having been priorized
                ns.flags |= flags.PRIORIZED

                # if the position wasn't fit then copy the errors
                # from the input catalog
                if stage < 2:
                    ns.err_ra = s.err_ra
                    ns.err_dec = s.err_dec
                    ns.flags |= flags.FIXED2PSF

                # if the shape wasn't fit then copy the errors
                # from the input catalog
                if stage < 3:
                    ns.err_a = s.err_a
                    ns.err_b = s.err_b
                    ns.err_pa = s.err_pa
            sources.extend(new_src)
        return sources

    def _fit_island(self, island_data):
        """
        Take an Island, do all the parameter estimation and fitting.


        Parameters
        ----------
        island_data : :class:`AegeanTools.models.IslandFittingData`
          The island to be fit.

        Returns
        -------
        sources : list
          The sources that were fit.
        """
        global_data = self.global_data

        # global data
        # dcurve = global_data.dcurve
        rmsimg = global_data.rmsimg

        # island data
        isle_num = island_data.isle_num
        idata = island_data.i
        innerclip, outerclip, max_summits = island_data.scalars
        xmin, xmax, ymin, ymax = island_data.offsets

        self.log.debug(
            "xmin xmax ymin ymax {0} {1} {2} {3}".format(
                xmin, xmax, ymin, ymax)
        )

        # get the beam parameters at the center of this island
        midra, middec = global_data.wcshelper.pix2sky(
            [0.5 * (xmax + xmin), 0.5 * (ymax + ymin)]
        )

        self.log.debug("midra middex {0} {1}".format(midra, middec))

        try:
            beam = global_data.psfhelper.get_psf_sky2pix(midra, middec)
        except ValueError:
            # This island has no psf or is not 'on' the sky, ignore it
            self.log.debug(
                "Beam sky2pix failed. Island has invalid WCS/Beam - Skipping."
            )
            return []

        del middec, midra

        # the curvature needs a buffer of 1 pixel to correctly
        # identify local min/max on the edge of the region.
        # We need a 1 pix buffer (if available)
        buffx = [
            xmin - max(xmin - 1, 0),
            min(xmax + 1, global_data.img.shape[0]) - xmax,
        ]
        buffy = [
            ymin - max(ymin - 1, 0),
            min(ymax + 1, global_data.img.shape[1]) - ymax,
        ]
        icurve = np.zeros(
            shape=(
                xmax - xmin + buffx[0] + buffx[1],
                ymax - ymin + buffy[0] + buffy[1],
            ),
            dtype=np.int8,
        )
        # compute peaks and convert to +/-1
        peaks = maximum_filter(
            self.global_data.img[
                xmin - buffx[0]: xmax + buffx[1],
                ymin - buffy[0]: ymax + buffy[0]],
            size=3,
        )
        pmask = np.where(
            peaks == self.global_data.img[
                xmin - buffx[0]: xmax + buffx[1],
                ymin - buffy[0]: ymax + buffy[0]]
        )
        troughs = minimum_filter(
            self.global_data.img[
                xmin - buffx[0]: xmax + buffx[1],
                ymin - buffy[0]: ymax + buffy[0]],
            size=3,
        )
        tmask = np.where(
            troughs == self.global_data.img[
                xmin - buffx[0]: xmax + buffx[1],
                ymin - buffy[0]: ymax + buffy[0]]
        )
        icurve[pmask] = -1
        icurve[tmask] = 1
        # icurve and idata need to be the same size so we crop
        # icurve based on the buffers that we computed
        icurve = icurve[
            buffx[0]: icurve.shape[0] - buffx[1],
            buffy[0]: icurve.shape[1] - buffy[1]]
        del peaks, pmask, troughs, tmask

        rms = rmsimg[xmin:xmax, ymin:ymax]

        is_flag = 0
        a, b, pa = global_data.psfhelper.get_psf_pix2pix(
            (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        )
        if not np.all(np.isfinite((a, b, pa))):
            # This island has no psf or is not 'on' the sky, ignore it
            self.log.debug("Island has invalid WCS/Beam - Skipping.")
            return []
        pixbeam = Beam(a, b, pa)

        self.log.debug("=====")
        self.log.debug("Island ({0})".format(isle_num))
        params = self.estimate_lmfit_parinfo(
            idata,
            rms,
            icurve,
            beam,
            innerclip,
            outerclip,
            offsets=[xmin, ymin],
            max_summits=max_summits,
        )
        # params = estimate_parinfo_image()
        # islands at the edge of a region of nans
        # result in no components
        if params is None or params["components"].value < 1:
            return []

        self.log.debug("Rms is {0}".format(np.shape(rms)))
        self.log.debug("Isle is {0}".format(np.shape(idata)))
        self.log.debug(
            " of which {0} are masked".format(sum(np.isnan(idata).ravel() * 1))
        )

        # Check that there is enough data to do the fit
        mx, my = np.where(np.isfinite(idata))
        non_blank_pix = len(mx)
        free_vars = len([1 for a in params.keys() if params[a].vary])
        if non_blank_pix < free_vars or free_vars == 0:
            self.log.debug(
                "Island {0} doesn't have enough pixels to fit the given model"
                .format(isle_num)
            )
            self.log.debug(
                "non_blank_pix {0}, free_vars {1}".format(
                    non_blank_pix, free_vars)
            )
            result = DummyLM()
            model = params
            is_flag |= flags.NOTFIT
        else:
            # Model is the fitted parameters
            fac = 1 / np.sqrt(2)
            if self.global_data.docov:
                C = Cmatrix(
                    mx,
                    my,
                    pixbeam.a * FWHM2CC * fac,
                    pixbeam.b * FWHM2CC * fac,
                    pixbeam.pa,
                )
                B = Bmatrix(C)
            else:
                C = B = None
            self.log.debug(
                "C({0},{1},{2},{3},{4})"
                .format(len(mx), len(my),
                        pixbeam.a * FWHM2CC,
                        pixbeam.b * FWHM2CC,
                        pixbeam.pa)
            )
            errs = np.nanmax(rms)
            self.log.debug("Initial params")
            self.log.debug(params)
            result, _ = do_lmfit(idata, params, B=B)
            if not result.errorbars:
                is_flag |= flags.FITERR
            # get the real (sky) parameter errors
            model = covar_errors(result.params, idata, errs=errs, B=B, C=C)

            if self.global_data.dobias and self.global_data.docov:
                x, y = np.indices(idata.shape)
                acf = elliptical_gaussian(
                    x, y, 1, 0, 0,
                    pixbeam.a * FWHM2CC * fac,
                    pixbeam.b * FWHM2CC * fac,
                    pixbeam.pa,
                )
                bias_correct(model, idata, acf=acf * errs ** 2)

            if not result.success:
                is_flag |= flags.FITERR

        self.log.debug("Final params")
        self.log.debug(model)

        # convert the fitting results to a list of sources [and islands]
        sources = self.result_to_components(
            result, model, island_data, is_flag)

        return sources

    def find_sources_in_image(
        self,
        filename,
        hdu_index=0,
        outfile=None,
        rms=None,
        bkg=None,
        max_summits=None,
        innerclip=5,
        outerclip=4,
        cores=None,
        rmsin=None,
        bkgin=None,
        beam=None,
        doislandflux=False,
        nopositive=False,
        nonegative=False,
        mask=None,
        imgpsf=None,
        blank=False,
        docov=True,
        cube_index=None,
        progress=False,
    ):
        """
        Run the Aegean source finder.


        Parameters
        ----------
        filename : str or HDUList
          Image filename or HDUList.

        hdu_index : int
          The index of the FITS HDU (extension).

        outfile : str
          file for printing catalog
          (NOT a table, just a text file of my own design)

        rms : float
          Use this rms for the entire image
          (will also assume that background is 0)

        max_summits : int
          Fit up to this many components to each island
          (extras are included but not fit)

        innerclip, outerclip : float
          The seed (inner) and flood (outer) clipping level (sigmas).

        cores : int
          Number of CPU cores to use. None means all cores.

        rmsin, bkgin : str or HDUList
          Filename or HDUList for the noise and background images.
          If either are None, then it will be calculated internally.

        beam : (major, minor, pa)
          Floats representing the synthesised beam (degrees).
          Replaces whatever is given in the FITS header.
          If the FITS header has no BMAJ/BMIN then this is required.

        doislandflux : bool
          If True then each island will also be characterized.

        nopositive, nonegative : bool
          Whether to return positive or negative sources.
          Default nopositive=False, nonegative=True.

        mask : str
          The filename of a region file created by MIMAS.
          Islands outside of this region will be ignored.

        imgpsf : str or HDUList
           Filename or HDUList for a psf image.

        blank : bool
          Cause the output image to be blanked where islands are found.

        docov : bool
          If True then include covariance matrix in the fitting process.
          Default=True

        cube_index : int
          For image cubes, cube_index determines which slice is used.

        progress : bool
          If true then show a progress bar when fitting island groups

        Returns
        -------
        sources : list
          List of sources found.
        """

        # Tell numpy to be quiet
        np.seterr(invalid="ignore")
        if cores is not None:
            if not (cores >= 1):
                raise AssertionError("cores must be one or more")
        else:
            cores = multiprocessing.cpu_count()

        self.load_globals(
            filename,
            hdu_index=hdu_index,
            bkgin=bkgin,
            rmsin=rmsin,
            beam=beam,
            verb=True,
            rms=rms,
            bkg=bkg,
            cores=cores,
            mask=mask,
            psf=imgpsf,
            blank=blank,
            docov=docov,
            cube_index=cube_index,
        )
        global_data = self.global_data
        rmsimg = global_data.rmsimg
        data = global_data.img

        self.log.info(
            "beam = {0:5.2f}'' x {1:5.2f}'' at {2:5.2f}deg".format(
                global_data.beam.a * 3600,
                global_data.beam.b * 3600,
                global_data.beam.pa,
            )
        )
        # stop people from doing silly things.
        if outerclip > innerclip:
            outerclip = innerclip
        self.log.info("seedclip={0}".format(innerclip))
        self.log.info("floodclip={0}".format(outerclip))

        islands = find_islands(
            im=data,
            bkg=np.zeros_like(data),
            rms=rmsimg,
            seed_clip=innerclip,
            flood_clip=outerclip,
            region=global_data.region,
            wcs=global_data.psfhelper,
            log=self.log,
        )
        self.log.info("Found {0} islands".format(len(islands)))
        self.log.info("Begin fitting")

        island_group = []
        isle_num = 0

        for island in islands:
            [[xmin, xmax], [ymin, ymax]] = island.bounding_box
            island_mask = island.mask

            i = copy.deepcopy(global_data.img[xmin:xmax, ymin:ymax])
            i[island_mask] = np.nan

            # ignore empty islands
            # This should now be impossible to trigger
            if not np.any(np.isfinite(i)):
                self.log.warning(
                    "Empty island detected, this should be imposisble.")
                continue
            isle_num += 1
            scalars = (innerclip, outerclip, max_summits)
            offsets = (xmin, xmax, ymin, ymax)
            island_data = IslandFittingData(
                isle_num, i, scalars, offsets, doislandflux)
            island_group.append(island_data)

        # now fit all the islands
        sources = []
        with tqdm(total=isle_num, desc="Fitting Islands:") as pbar:
            for i in island_group:
                try:
                    pbar.update(1)
                    srcs = self._fit_island(i)
                except AegeanNaNModelError:
                    continue
                # update bar as each individual island is fit
                for src in srcs:
                    # ignore sources that we have been told to ignore
                    if (src.peak_flux > 0 and nopositive) or (
                        src.peak_flux < 0 and nonegative
                    ):
                        continue
                    sources.append(src)

        # Write the output to the output file
        if outfile:
            print(
                header.format(
                    "{0}-({1})".format(__version__, __date__), filename),
                file=outfile,
            )
            print(ComponentSource.header, file=outfile)
            for s in sources:
                print(str(s), file=outfile)

        self.sources.extend(sources)
        self.log.info("Fit {0} sources".format(len(sources)))
        return sources

    def priorized_fit_islands(
        self,
        filename,
        catalogue,
        hdu_index=0,
        outfile=None,
        bkgin=None,
        rmsin=None,
        cores=1,
        rms=None,
        bkg=None,
        beam=None,
        imgpsf=None,
        catpsf=None,
        stage=3,
        ratio=None,
        outerclip=3,
        doregroup=True,
        regroup_eps=None,
        docov=True,
        cube_index=None,
        progress=False,
    ):
        """
        Take an input catalog, and image, and optional background/noise images
        fit the flux and ra/dec for each of the given sources, keeping the
        morphology fixed

        Multiple cores can be specified, and will be used.

        Parameters
        ----------
        filename : str or HDUList
            Image filename or HDUList.

        catalogue : str or list
            Input catalogue file name or list of ComponentSource objects.

        hdu_index : int
            The index of the FITS HDU (extension).

        outfile : str
            file for printing catalog
            (NOT a table, just a text file of my own design)

        rmsin, bkgin : str or HDUList
            Filename or HDUList for the noise and background images.
            If either are None, then it will be calculated internally.

        cores : int
            Number of CPU cores to use. None means all cores.

        rms : float
            Use this rms for the entire image
            (will also assume that background is 0)

        beam : (float, float, float)
            (major, minor, pa) representing the synthesised beam (degrees).
            Replaces whatever is given in the FITS header.
            If the FITS header has no BMAJ/BMIN then this is required.

        imgpsf : str or HDUList
            Filename or HDUList for a psf image.

        catpsf : str or HDUList
            Filename or HDUList for the catalogue psf image.

        stage : int
            Refitting stage

        ratio : float
            If not None - ratio of image psf to catalog psf,
            otherwise interpret from catalogue or image if possible

        outerclip : float
            The flood (outer) clipping level (sigmas).

        doregroup : bool, Default=True
            Relabel all the islands/groups to ensure that nearby
            components are jointly fit.

        regroup_eps: float, Default=None
            The linking parameter for regouping. Components that are
            closer than this distance (in arcmin) will be jointly fit.
            If NONE, then use 4x the average source major axis size (after
            rescaling if required).

        docov : bool, Default=True
            If True then include covariance matrix in the fitting process.

        cube_index : int, Default=None
            For image cubes, slice determines which slice is used.

        progress : bool, Default=True
            Show a progress bar when fitting island groups

        Returns
        -------
        sources : list
            List of sources measured.

        """

        from AegeanTools.cluster import regroup_dbscan

        self.load_globals(
            filename,
            hdu_index=hdu_index,
            bkgin=bkgin,
            rmsin=rmsin,
            beam=beam,
            verb=True,
            rms=rms,
            bkg=bkg,
            cores=cores,
            do_curve=False,
            psf=imgpsf,
            docov=docov,
            cube_index=cube_index,
        )

        global_data = self.global_data

        # load the table and convert to an input source list
        if isinstance(catalogue, str):
            input_table = load_table(catalogue)
            input_sources = np.array(table_to_source_list(input_table))
        else:
            input_sources = np.array(catalogue)

        if len(input_sources) < 1:
            self.log.debug("No input sources for priorized fitting")
            return []

        # reject sources with missing params
        ok = True
        for param in ["ra", "dec", "peak_flux", "a", "b", "pa"]:
            if np.isnan(getattr(input_sources[0], param)):
                self.log.info("Source 0, is missing param '{0}'".format(param))
                ok = False
        if not ok:
            self.log.error("Missing parameters! Not fitting.")
            self.log.error(
                "Maybe your table is missing or mis-labeled columns?")
            return []
        del ok

        # Do the resizing
        self.log.info("{0} sources in catalog".format(len(input_sources)))
        sources = cluster.resize(
            input_sources, ratio=ratio, psfhelper=global_data.psfhelper)
        self.log.info("{0} sources accepted".format(len(sources)))

        if len(sources) < 1:
            self.log.debug("No sources accepted for priorized fitting")
            return []

        # compute eps if it's not defined
        if regroup_eps is None:
            # s.a is in arcsec but we assume regroup_eps is in arcmin
            regroup_eps = 4*np.mean([s.a/60 for s in sources])
        # convert regroup_eps into a value appropriate for a cartesian measure
        regroup_eps = np.sin(np.radians(regroup_eps/60))
        input_sources = sources
        # redo the grouping if required
        if doregroup:
            # TODO: scale eps in some appropriate manner
            groups = regroup_dbscan(input_sources, eps=regroup_eps)
        else:
            groups = list(island_itergen(input_sources))

        self.log.info("Begin fitting")

        island_groups = []  # will be a list of groups of islands
        island_group = []  # will be a list of islands
        group_size = 20

        for island in groups:
            island_group.append(island)
            # If the island group is full queue it for the subprocesses to fit
            if len(island_group) >= group_size:
                island_groups.append(island_group)
                island_group = []
        # The last partially-filled island group also needs to
        #  be queued for fitting
        if len(island_group) > 0:
            island_groups.append(island_group)

        sources = []
        with tqdm(
            total=len(island_groups),
            desc="Refitting Island Groups",
            disable=not progress,
        ) as pbar:
            for i, g in enumerate(island_groups):
                srcs = self._refit_islands(g, stage, outerclip, istart=i)
                # update bar as each individual island is fit
                pbar.update(1)
                sources.extend(srcs)

        sources = sorted(sources)

        # Write the output to the output file
        if outfile:
            print(
                header.format(
                    "{0}-({1})".format(__version__, __date__), filename),
                file=outfile,
            )
            print(ComponentSource.header, file=outfile)
            for source in sources:
                print(str(source), file=outfile)

        self.log.info("fit {0} components".format(len(sources)))
        self.sources.extend(sources)
        return sources


# Helpers
def fix_shape(source):
    """
    Ensure that a>=b for a given source object.
    If a<b then swap a/b and increment pa by 90.
    err_a/err_b are also swapped as needed.

    Parameters
    ----------
    source : object
      any object with a/b/pa/err_a/err_b properties

    """
    if source.a < source.b:
        source.a, source.b = source.b, source.a
        source.err_a, source.err_b = source.err_b, source.err_a
        source.pa += 90
    return


def pa_limit(pa):
    """
    Position angle is periodic with period 180 deg
    Constrain pa such that -90<pa<=90

    Parameters
    ----------
    pa : float
      Initial position angle.

    Returns
    -------
    pa : float
      Rotate position angle.
    """
    while pa <= -90:
        pa += 180
    while pa > 90:
        pa -= 180
    return pa


def theta_limit(theta):
    """
    Angle theta is periodic with period pi.
    Constrain theta such that -pi/2<theta<=pi/2.

    Parameters
    ----------
    theta : float
      Input angle.

    Returns
    -------
    theta : float
      Rotate angle.
    """
    while theta <= -1 * np.pi / 2:
        theta += np.pi
    while theta > np.pi / 2:
        theta -= np.pi
    return theta


def get_aux_files(basename):
    """
    Look for and return all the aux files that are associated with
    this filename. Will look for:
    - background (_bkg.fits)
    - rms        (_rms.fits)
    - mask       (.mim)
    - catalogue  (_comp.fits)
    - psf map    (_psf.fits)

    will return filenames if they exist, or None where they do not.

    Parameters
    ----------
    basename : str
      The name/path of the input image.

    Returns
    -------
    aux : dict
      Dict of filenames or None with keys (bkg, rms, mask, cat, psf)
    """
    base = os.path.splitext(basename)[0]
    files = {
        "bkg": base + "_bkg.fits",
        "rms": base + "_rms.fits",
        "mask": base + ".mim",
        "cat": base + "_comp.fits",
        "psf": base + "_psf.fits",
    }

    for k in files.keys():
        if not os.path.exists(files[k]):
            files[k] = None
    return files
