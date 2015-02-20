#! /usr/bin/env python

"""
Test the errors reported by Aegean.
"""
__author__="Paul Hancock"

import sys
sys.path.append("../.")

import aegean as ae
from astropy.io import fits
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor
from AegeanTools.mpfit import mpfit
from scipy.linalg import sqrtm
import logging



def multi_gauss(data, rmsimg, parinfo):
    """
    Fit multiple gaussian components to data using the information provided by parinfo.
    data may contain 'flagged' or 'masked' data with the value of np.NaN
    input: data - pixel information
           rmsimg - image containing 1sigma values
           parinfo - initial parameters for mpfit
    return: mpfit object, parameter info
    """
    shape = data.shape
    coords = np.where(np.isfinite(data))
    data = np.ravel(data)
    mask = np.where(np.isfinite(data))  #the indices of the *non* NaN values in data
    data = data[mask]

    z = np.ravel(ae.ntwodgaussian([1, 1, 1, smoothing, smoothing, 0])(*coords))
    C = np.vstack((z,)*len(z))
    for i in range(len(z)):
        C[i] = np.roll(C[i],i)
    Ci = np.matrix(C).I
    #Ci = np.matrix(np.diag(np.ones((len(z)))))
    #Ci = sqrtm(Ci)
    def model(p):
        """Return a map with a number of Gaussians determined by the input parameters."""
        return np.ravel(ae.ntwodgaussian(p)(*coords))


    def erfunc(p, fjac=None):
        """The difference between the model and the data"""
        resid = (model(p)-data) * Ci # * np.matrix(model-data).T
        resid =  np.array(resid.tolist()[0])
        #return [0, model(p) - data ]
        return [0, resid]

    mp = mpfit(erfunc, parinfo=parinfo, quiet=True)
    mp.dof = len(data) - len(parinfo)
    return mp, parinfo


# sub the new version in
ae.multi_gauss = multi_gauss

# load file to get wcs
hdulist = fits.open('Images/1904-66_SIN.fits')

shape = hdulist[0].data.shape
# create boring background/rms
background = np.zeros(shape)
rms = np.ones(shape)

# figure out the smoothing for our noise
scale = abs(hdulist[0].header['CDELT1']*hdulist[0].header['CDELT2'])
smoothing = hdulist[0].header['BMAJ'] * hdulist[0].header['BMIN'] / scale
smoothing = np.sqrt(smoothing) * ae.fwhm2cc
print smoothing

# create some correlated noise
np.random.seed(1234567890)
noise = np.random.random(shape)
noise = gaussian_filter(noise, sigma=smoothing)
noise -= np.mean(noise) # zero mean
noise /= np.std(noise) # unit rms

# make a bunch of sources within the image
stepsize = smoothing * 10
xlocs = np.arange(int(floor(stepsize)), shape[0], int(floor(stepsize)))
ylocs = np.arange(int(floor(stepsize)), shape[1], int(floor(stepsize)))

fluxes = []
rapix = []
decpix = []
signal = np.zeros(shape)
xos, yos = np.meshgrid(xlocs,ylocs)

xpix, ypix = np.meshgrid(range(shape[0]),range(shape[1]))
for f, loc in enumerate(zip(np.ravel(xos),np.ravel(yos))):
    flux = 5+f*2
    xo, yo = loc
    major, minor = smoothing, smoothing
    pa = 0
    src = ae.ntwodgaussian([flux, xo, yo, major, minor, pa])(xpix,ypix)
    fluxes.append(flux)
    rapix.append(xo)
    decpix.append(yo)
    signal += src

hdulist[0].data = signal + noise

hdulist.writeto('test.fits',clobber=True)

sources = ae.find_sources_in_image(hdulist, max_summits=2, rms=1, cores=1)
print "found {0}/{1} sources".format(len(sources),f+1)
ae.save_catalog('test_m.vot',sources)
ae.save_catalog('test_m.reg',sources)


out = open('test.csv','w')
print >>out, ','.join(['flux', 'ra', 'dec', 'major', 'minor', 'pa'])
for f, xo, yo in zip(fluxes, rapix, decpix):
    major, minor = smoothing, smoothing
    pa = 0
    ra, dec = ae.pix2sky([yo+1,xo+1]) # not sure why I need +1 here
    print xo,yo,"->",ra,dec
    print >>out, ','.join(map(str,[f, ra, dec, major*np.sqrt(scale)*ae.cc2fwhm *3600, minor*np.sqrt(scale)*ae.cc2fwhm *3600, pa]))
out.close
# now compare the various parameters to the known values.

