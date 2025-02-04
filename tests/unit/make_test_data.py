#! /usr/bin/env python
from copy import deepcopy

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from AegeanTools import AeRes, models
from AegeanTools.logging import logger, logging
from AegeanTools.source_finder import FWHM2CC
from AegeanTools.wcs_helpers import WCSHelper
from AegeanTools.catalogs import write_catalog

__author__ = "Paul Hancock"

imsize = (8, 256, 512)  # non square picks up more errors
noise = 0.5  # non unity noise picks up more errors
psf = [30, 30, 0]  # psf in arcsec,arcsec,degrees
pix_per_beam = 3.5
seed = 123987554
ra, dec = (30, -15)
nu0 = 162 * 1e6


def make_noise_image(as_cube=False):
    """
    Create a noise-only image and a WCS object for use in testing.

    returns
    -------
    image : :class:`np.ndarray`
        An image of noise.

    wcs : :class:`astropy.wcs.WCS`
        A wcs object that is valid for this image.
    """

    np.random.seed(seed)
    image = np.random.random(size=imsize)

    for i in range(image.shape[0]):
        image[i] = gaussian_filter(image[i], sigma=pix_per_beam * FWHM2CC)
        # force zero mean and unit variance
        image[i] -= np.mean(image[i])
        image[i] /= np.std(image[i])
        image[i] *= noise

    image = np.array(image, dtype=np.float32)

    if as_cube:
        wcs = WCS(naxis=3)
        wcs.wcs.crpix = [imsize[1] / 2, imsize[2] / 2, 0]
        wcs.wcs.cdelt = np.array(
            [psf[0] / 3600 / pix_per_beam, psf[1] / 3600 / pix_per_beam, 1e6]
        )
        wcs.wcs.crval = [ra, dec, nu0]
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ"]
    else:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [imsize[1] / 2, imsize[2] / 2]
        wcs.wcs.cdelt = np.array(
            [psf[0] / 3600 / pix_per_beam, psf[1] / 3600 / pix_per_beam]
        )
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        image = image[0]
    return image, wcs


def make_psf_map(base):
    """
    Create a small psf map for use in testing.
    """

    psf_data = np.ones((3, imsize[1], imsize[2]))
    psf_data *= np.array(psf)[:, None, None] / 3600.0  # acrsec -> degrees
    # create a few 'holes' in the map
    psf_data[:, 56, 34] = np.nan

    # blank one edge of the map
    psf_data[:, :, 0] = np.nan
    hdu = deepcopy(base)
    hdu[0].data = psf_data
    hdu[0].header["CTYPE3"] = ("Beam", "0=a,1=b,2=pa (degrees)")
    return hdu


def make_catalogue():
    """
    Create a catalogue of sources which will be found
    """
    cat = []
    # creage a grid of point sources with SNR from 1-100
    ras = np.linspace(ra - 0.2, ra + 0.8, 10)
    decs = np.linspace(dec - 0.5, dec - 0.1, 10)
    fluxes = [(a + 1) * noise for a in range(100)]

    for i, r in enumerate(ras):
        for j, d in enumerate(decs):
            src = models.ComponentSource()
            src.ra = r
            src.dec = d
            src.ra_str = "00:00:00"
            src.dec_str = "-00:00:00"
            src.a, src.b, src.pa = psf
            src.island = i * 10 + j
            src.peak_flux = fluxes[i * 10 + j]
            src.background = 0
            src.local_rms = noise
            src.err_peak_flux = noise
            cat.append(src)

    # create an island from a diffuse source
    src = models.ComponentSource()
    src.island = 100
    src.ra = 30.88
    src.dec = -15.04
    src.ra_str = "00:00:00"
    src.dec_str = "-00:00:00"
    src.a, src.b, src.pa = psf
    src.a *= 3
    src.b *= 3
    src.peak_flux = 6 * noise
    cat.append(src)

    # ensure that one of the islands has two sources in it
    src = models.ComponentSource()
    src.ra = 30.78
    src.dec = -15.102
    src.a, src.b, src.pa = psf
    src.ra_str = "00:00:00"
    src.dec_str = "-00:00:00"
    src.island = 101
    src.peak_flux = 80 * noise
    cat.append(src)

    return cat


if __name__ == "__main__":
    # configure logging
    logging_level = logging.INFO
    logger.setLevel(logging_level)

    # make 3d noise image
    image, wcs = make_noise_image(as_cube=True)
    logger.info(f"Image shape {image.shape}")
    header = wcs.to_header()
    header["BMAJ"] = psf[0] / 3600
    header["BMIN"] = psf[1] / 3600
    header["BPA"] = psf[2]
    hdu = fits.HDUList(hdus=[fits.PrimaryHDU(header=header, data=image)])

    # make catalogue without alpha
    cat_no_alpha = make_catalogue()

    # make catalogue with alpha
    # cat_with_alpha = deepcopy(cat_no_alpha)
    cat_with_alpha = [
        models.ComponentSource3D.from_component_source(c) for c in cat_no_alpha
    ]
    for c in cat_with_alpha:
        c.alpha = np.random.uniform(-2, 1)
        c.nu0 = nu0

    # save the image cube, catalogue, and psf map
    wcshelper = WCSHelper.from_header(hdu[0].header)
    hdu[0].data += AeRes.make_model(cat_with_alpha, image.shape, wcshelper)
    hdu.writeto("synthetic_cube.fits", overwrite=True)
    psf_hdu = make_psf_map(hdu)
    psf_hdu.writeto("synthetic_cube_psf.fits", overwrite=True)
    write_catalog("synthetic_cat_with_alpha.fits", cat_with_alpha, fmt="fits")

    # make 2d image and save
    image, wcs = make_noise_image(as_cube=False)
    header = wcs.to_header()
    header["BMAJ"] = psf[0] / 3600
    header["BMIN"] = psf[1] / 3600
    header["BPA"] = psf[2]
    hdu = fits.HDUList(hdus=[fits.PrimaryHDU(header=header, data=image)])

    wcshelper = WCSHelper.from_header(hdu[0].header)
    hdu[0].data += AeRes.make_model(cat_no_alpha, (1, *image.shape), wcshelper)[0, :, :]
    hdu.writeto("synthetic_image.fits", overwrite=True)
    psf_hdu = make_psf_map(hdu)
    psf_hdu.writeto("synthetic_image_psf.fits", overwrite=True)
    write_catalog("synthetic_cat_no_alpha.fits", cat_no_alpha, fmt="fits")
