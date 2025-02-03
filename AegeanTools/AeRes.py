#! /usr/bin/env python
"""
Aegean Residual (AeRes) has the following capability:
- convert a catalogue into an image model
- subtract image model from image
- write model and residual files
"""


import numpy as np
from astropy.io import fits

from AegeanTools import catalogs, fitting, wcs_helpers
from AegeanTools.logging import logger, logging
from AegeanTools.models import ComponentSource, ComponentSource3D
from AegeanTools.fits_tools import load_image_band, write_fits

__author__ = "Paul Hancock"

FWHM2CC = 1 / (2 * np.sqrt(2 * np.log(2)))


def load_sources(
    filename,
    ra_col="ra",
    dec_col="dec",
    peak_col="peak_flux",
    a_col="a",
    b_col="b",
    pa_col="pa",
    alpha_col="alpha",
    nu0_col="nu0",
):
    """
    Open a file, read contents, return a list of all the sources in that file.

    Parameters
    ----------
    filename : str
        Filename to be read

    ra_col, dec_col, peak_col, a_col, b_col, pa_col : str
        The column names for each of the parameters.
        Default = ['ra', 'dec', 'peak_flux', 'a', 'b', 'pa', 'alpha' ,'nu0']

    Return
    ------
    catalog : [`class:AegeanTools.models.ComponentSource`, ...]
        A list of source components
    """
    srctype = ComponentSource
    table = catalogs.load_table(filename)
    required_cols = [ra_col, dec_col, peak_col, a_col, b_col, pa_col]
    # required_cols = ['ra','dec','peak_flux','a','b','pa']
    good = True
    for c in required_cols:
        if c not in table.colnames:
            logger.error("Column {0} not found".format(c))
            good = False
    if not good:
        logger.error("Some required columns missing or mis-labeled")
        return None
    # rename the table columns
    for old, new in zip(
        [ra_col, dec_col, peak_col, a_col, b_col, pa_col],
        ["ra", "dec", "peak_flux", "a", "b", "pa"],
    ):
        table.rename_column(old, new)

    # rename the alpha column if it exists
    if alpha_col in table.colnames:
        table.rename_column(alpha_col, "alpha")
        table.rename_column(nu0_col, "nu0")
        srctype = ComponentSource3D

    catalog = catalogs.table_to_source_list(table, src_type=srctype)
    logger.info("read {0} sources from {1}".format(len(catalog), filename))
    return catalog


def make_model(sources, shape, wcshelper, mask=False, frac=None, sigma=4):
    """
    Create a model image based on a catalogue of sources.

    Parameters
    ----------
    sources : [`class:AegeanTools.models.ComponentSource`, ...]
        a list of sources

    shape : [int, int, int]
        the shape of the input (and output) image

    wcshelper : 'class:AegeanTools.wcs_helpers.WCSHelper'
        A WCSHelper object corresponding to the input image

    mask : bool
        If true then mask pixels instead of subtracting or adding sources

    frac : float
        pixels that are brighter than frac*peak_flux for each source will be
        masked if mask=True

    sigma: float
        pixels that are brighter than rms*sigma be masked if mask=True

    Returns
    -------
    model : np.ndarray
        The desired model.
    """

    # Model array
    m = np.zeros(shape, dtype=np.float32)
    factor = 5

    i_count = 0
    for s in range(shape[0]):
        for src in sources:
            xo, yo, sx, sy, theta = wcshelper.sky2pix_ellipse(
                [src.ra, src.dec], src.a / 3600, src.b / 3600, src.pa
            )
            phi = np.radians(theta)

            # skip sources that have a center that is outside of the image
            if not 0 < xo < shape[1]:
                logger.debug("source {0} is not within image".format(src.island))
                continue
            if not 0 < yo < shape[2]:
                logger.debug("source {0} is not within image".format(src.island))
                continue

            # pixels over which this model is calculated
            xoff = factor * (abs(sx * np.cos(phi)) + abs(sy * np.sin(phi)))
            xmin = xo - xoff
            xmax = xo + xoff

            yoff = factor * (abs(sx * np.sin(phi)) + abs(sy * np.cos(phi)))
            ymin = yo - yoff
            ymax = yo + yoff

            # clip to the image size
            ymin = max(np.floor(ymin), 0)
            ymax = min(np.ceil(ymax), shape[2])

            xmin = max(np.floor(xmin), 0)
            xmax = min(np.ceil(xmax), shape[1])

            if not np.all(np.isfinite([ymin, ymax, xmin, xmax])):
                continue

            # Compute the peak flux at this frequency
            peak_flux = src.peak_flux
            if hasattr(src, "nu0"):
                peak_flux *= (wcshelper.pix2freq(s) / src.nu0) ** src.alpha

            if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
                logger.debug("Source ({0},{1})".format(src.island, src.source))
                logger.debug(" xo, yo: {0} {1}".format(xo, yo))
                logger.debug(" sx, sy: {0} {1}".format(sx, sy))
                logger.debug(" theta, phi: {0} {1}".format(theta, phi))
                logger.debug(" xoff, yoff: {0} {1}".format(xoff, yoff))
                logger.debug(
                    " xmin, xmax, ymin, ymax: {0}:{1} {2}:{3}".format(
                        xmin, xmax, ymin, ymax
                    )
                )
                logger.debug(" flux, sx, sy: {0} {1} {2}".format(peak_flux, sx, sy))

            # positions for which we want to make the model
            x, y = np.mgrid[int(xmin) : int(xmax), int(ymin) : int(ymax)]
            x = x.ravel()
            y = y.ravel()

            # TODO: understand why xo/yo -1 is needed
            model = fitting.elliptical_gaussian(
                x, y, src.peak_flux, xo - 1, yo - 1, sx * FWHM2CC, sy * FWHM2CC, theta
            )

            # Mask the output image if requested
            if mask:
                if frac is not None:
                    indices = np.where(model >= (frac * peak_flux))
                else:
                    indices = np.where(model >= (sigma * src.local_rms))
                # somehow m[x,y][indices] = np.nan doesn't assign any values
                # so we have to do the more complicated
                # m[x[indices],y[indices]] = np.nan
                m[s, x[indices], y[indices]] = np.nan
            else:
                m[s, x, y] += model

            if s == 0:
                i_count += 1
    logger.info("modeled {0} sources".format(i_count))
    return m


def make_residual(
    fitsfile,
    catalog,
    rfile,
    mfile=None,
    add=False,
    mask=False,
    frac=None,
    sigma=4,
    colmap=None,
):
    """
    Take an input image and catalogue, make a model of the catalogue, and then
    add/subtract or mask the input image. Saving the residual and (optionally)
    model files.

    Parameters
    ----------
    fitsfile : str
        Input fits image filename

    catalog : str
        Input catalog filename of a type supported by Aegean

    rfile : str
        Filename to write residual image

    mfile : str
        Filename to write model image. Default=None means don't write the model
        file.

    add : bool
        If True add the model instead of subtracting it

    mask : bool
        If true then mask pixels instead of adding or subtracting the sources

    frac : float
        pixels that are brighter than frac*peak_flux for each source will be
        masked if mask=True

    sigma : float
        pixels that are brighter than sigma*local_rms for each source will be
        masked if mask=True

    colmap : dict
        A mapping of column names. Default is: {'ra_col':'ra', 'dec_col':'dec',
        'peak_col':'peak_flux', 'a_col':'a', 'b_col':'b', 'pa_col':'pa, 'alpha_col':'alpha'}

    Return
    ------
    None
    """

    if colmap is None:
        colmap = {}

    source_list = load_sources(catalog, **colmap)

    if source_list is None:
        return None

    data, header = load_image_band(fitsfile, as_cube=True)

    wcshelper = wcs_helpers.WCSHelper.from_header(header)

    model = make_model(source_list, data.shape, wcshelper, mask, frac, sigma)

    if add or mask:
        residual = data + model
    else:
        residual = data - model

    write_fits(residual, header, rfile)
    if mfile is not None:
        write_fits(model, header, mfile)

    return
