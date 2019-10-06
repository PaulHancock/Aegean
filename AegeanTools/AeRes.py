#! /usr/bin/env python
"""
Aegean Residual (AeRes) has the following capability:
- convert a catalogue into an image model
- subtract image model from image
- write model and residual files
"""

__author__ = "Paul Hancock"

import logging
import numpy as np
from astropy.io import fits
from AegeanTools import catalogs, fitting, wcs_helpers

FWHM2CC = 1 / (2 * np.sqrt(2 * np.log(2)))


def load_sources(filename):
    """
    Open a file, read contents, return a list of all the sources in that file.
    @param filename:
    @return: list of ComponentSource objects
    """
    catalog = catalogs.table_to_source_list(catalogs.load_table(filename))
    logging.info("read {0} sources from {1}".format(len(catalog), filename))
    return catalog


def make_model(sources, shape, wcshelper, mask=False, frac=None, sigma=4):
    """

    @param sources: a list of AegeanTools.models.SimpleSource objects
    @param shape: the shape of the input (and output) image
    @param wcshelper: an AegeanTools.wcs_helpers.WCSHelper object corresponding to the input image
    @param mask: If true then mask pixels instead of subtracting the sources
    @param frac: pixels that are brighter than frac*peak_flux for each source will be masked if mask=True
    @param sigma: pixels that are brighter than rms*sigma be masked if mask=True
    @return:
    """

    # Model array
    m = np.zeros(shape, dtype=np.float32)
    factor = 5

    i_count = 0
    for src in sources:
        xo, yo, sx, sy, theta = wcshelper.sky2pix_ellipse([src.ra, src.dec], src.a/3600, src.b/3600, src.pa)
        phi = np.radians(theta)

        # skip sources that have a center that is outside of the image
        if not 0 < xo < shape[0]:
            logging.debug("source {0} is not within image".format(src.island))
            continue
        if not 0 < yo < shape[1]:
            logging.debug("source {0} is not within image".format(src.island))
            continue

        # pixels over which this model is calculated
        xoff = factor*(abs(sx*np.cos(phi)) + abs(sy*np.sin(phi)))
        xmin = xo - xoff
        xmax = xo + xoff

        yoff = factor*(abs(sx*np.sin(phi)) + abs(sy*np.cos(phi)))
        ymin = yo - yoff
        ymax = yo + yoff

        # clip to the image size
        ymin = max(np.floor(ymin), 0)
        ymax = min(np.ceil(ymax), shape[1])

        xmin = max(np.floor(xmin), 0)
        xmax = min(np.ceil(xmax), shape[0])

        if not np.all(np.isfinite([ymin, ymax, xmin, xmax])):
            continue

        if logging.getLogger().isEnabledFor(logging.DEBUG):  # pragma: no cover
            logging.debug("Source ({0},{1})".format(src.island, src.source))
            logging.debug(" xo, yo: {0} {1}".format(xo, yo))
            logging.debug(" sx, sy: {0} {1}".format(sx, sy))
            logging.debug(" theta, phi: {0} {1}".format(theta, phi))
            logging.debug(" xoff, yoff: {0} {1}".format(xoff, yoff))
            logging.debug(" xmin, xmax, ymin, ymax: {0}:{1} {2}:{3}".format(xmin, xmax, ymin, ymax))
            logging.debug(" flux, sx, sy: {0} {1} {2}".format(src.peak_flux, sx, sy))

        # positions for which we want to make the model
        x, y = np.mgrid[xmin:xmax, ymin:ymax]
        x = list(map(int, x.ravel()))
        y = list(map(int, y.ravel()))

        # TODO: understand why xo/yo -1 is needed
        model = fitting.elliptical_gaussian(x, y, src.peak_flux, xo-1, yo-1, sx*FWHM2CC, sy*FWHM2CC, theta)

        # Mask the output image if requested
        if mask:
            if frac is not None:
                indices = np.where(model >= (frac*src.peak_flux))
            else:
                indices = np.where(model >= (sigma*src.local_rms))
            model[indices] = np.nan

        m[x, y] += model
        i_count += 1
    logging.info("modeled {0} sources".format(i_count))
    return m


def make_residual(fitsfile, catalog, rfile, mfile=None, add=False, mask=False, frac=None, sigma=4):
    """

    @param fitsfile: Input fits image filename
    @param catalog: Input catalog filename of a type supported by Aegean
    @param rfile: Residual image filename
    @param mfile: model image filename
    @param add: add the model instead of subtracting it
    @param mask: If true then mask pixels instead of subtracting the sources
    @param frac: pixels that are brighter than frac*peak_flux for each source will be masked if mask=True
    @return:
    """
    source_list = load_sources(catalog)
    # force two axes so that we dump redundant stokes/freq axes if they are present.
    hdulist = fits.open(fitsfile, naxis=2)
    # ignore dimensions of length 1
    data = np.squeeze(hdulist[0].data)
    header = hdulist[0].header

    wcshelper = wcs_helpers.WCSHelper.from_header(header)

    model = make_model(source_list, data.shape, wcshelper, mask, frac, sigma)

    if add or mask:
        residual = data + model
    else:
        residual = data - model

    hdulist[0].data = residual
    hdulist.writeto(rfile, overwrite=True)
    logging.info("wrote residual to {0}".format(rfile))
    if mfile is not None:
        hdulist[0].data = model
        hdulist.writeto(mfile, overwrite=True)
        logging.info("wrote model to {0}".format(mfile))
    return
