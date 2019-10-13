#! /usr/bin/env python
from __future__ import print_function, division

__author__ = 'Paul Hancock'

from AegeanTools import models, AeRes
from AegeanTools.source_finder import FWHM2CC
from AegeanTools.wcs_helpers import WCSHelper
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.wcs import WCS
from astropy.io import fits
import logging

imsize = (256, 512) # non square picks up more errors
noise = 0.5 # non unity noise picks up more errors
psf = [30, 30, 0] # psf in arcsec,arcsec,degrees
pix_per_beam = 3.5
seed = 123987554
ra,dec = (30, -15)

def make_noise_image():
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
    image = np.random.random(size=imsize)*noise
    image = gaussian_filter(image, sigma=pix_per_beam*FWHM2CC)
    image = np.array(image, dtype=np.float32)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [imsize[0]/2, imsize[1]/2]
    wcs.wcs.cdelt = np.array([psf[0]/3600/pix_per_beam, psf[1]/3600/pix_per_beam])
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return image, wcs


def make_psf_map():
    """
    Create a small psf map for use in testing.
    """


def make_catalogue():
    """
    Create a catalogue of sources which will be found
    """
    cat = []
    # creage a grid of point sources with SNR from 1-100
    ras = np.linspace(ra-0.2, ra+0.8, 10)
    decs = np.linspace(dec-0.5, dec-0.1, 10)
    print(ras)
    print(decs)
    fluxes = [(a+1)*noise for a in range(100)]
    for i, r in enumerate(ras):
        for j, d in enumerate(decs):
            src = models.SimpleSource()
            src.ra = r
            src.dec = d
            src.a, src.b, src.pa = psf
            src.island = i*10 + j
            src.peak_flux = fluxes[i*10 +j]
            src.background = 0
            src.local_rms = noise
            src.err_peak_flux = noise
            cat.append(src)

    # create an island from a diffuse source
    src = models.SimpleSource()
    src.ra = 30.88
    src.dec = -15.04
    src.a, src.b, src.pa = psf
    src.a *= 3
    src.b *= 3
    src.island = 101
    src.peak_flux = 6*noise
    cat.append(src)

    # ensure that one of the islands has two sources in it
    src = models.SimpleSource()
    src.ra = 30.78
    src.dec = -15.102
    src.a, src.b, src.pa = psf
    src.island = 102
    src.peak_flux = 80*noise
    cat.append(src)

    return cat

def add_catalogue_image(catalogue, image):
    """

    """
    for c in catalogue:
        image += AeRes.make_model()

if __name__=="__main__":
    # configure logging
    logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
    log = logging.getLogger("Aegean")
    logging_level = logging.DEBUG # if options.debug else logging.INFO
    log.setLevel(logging_level)

    image, wcs = make_noise_image()
    header = wcs.to_header()
    header['BMAJ'] = psf[0]/3600
    header['BMIN'] = psf[1]/3600
    header['BPA'] = psf[2]
    hdu = fits.HDUList(hdus=[fits.PrimaryHDU(header=header, data=image)])
    header = hdu[0].header

    cat = make_catalogue()
    wcshelper = WCSHelper.from_header(header)
    hdu[0].data += AeRes.make_model(cat, image.shape, wcshelper)

    hdu.writeto('synthetic_test.fits', overwrite=True)