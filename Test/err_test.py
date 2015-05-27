#! /usr/bin/env python

"""
Test the errors reported by Aegean.
"""
__author__="Paul Hancock"

import sys
sys.path.append("../.")

import aegean as ae
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor
from matplotlib import pyplot
import lmfit

smoothing = 1

### OLD CODE THAT NO LONGER WORKS WITH LMFIT VERSION OF AEGEAN ###
def new_multi_gauss(data, rmsimg, parinfo):
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

def test_corr_noise_error():
    # sub the new version in
    ae.multi_gauss = new_multi_gauss

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
    out.close()

def dk_error_comp():
    global smoothing

    print "loading data"
    # load file to get wcs
    hdulist = fits.open('Images/1904-66_SIN.fits')

    shape = hdulist[0].data.shape

    # figure out the smoothing for our noise
    scale = abs(hdulist[0].header['CDELT1']*hdulist[0].header['CDELT2'])
    smoothing = hdulist[0].header['BMAJ'] * hdulist[0].header['BMIN'] / scale
    smoothing = np.sqrt(smoothing) * ae.fwhm2cc
    print smoothing

    print "making noises"
    # create some correlated noise
    np.random.seed(1234567890)
    noise = np.random.random(shape)
    cnoise = gaussian_filter(noise, sigma=smoothing)
    cnoise -= np.mean(cnoise) # zero mean
    cnoise /= np.std(cnoise) # unit rms

    # make a bunch of sources within the image
    stepsize = smoothing * 10
    xlocs = np.arange(int(floor(stepsize)), shape[0], int(floor(stepsize)))
    ylocs = np.arange(int(floor(stepsize)), shape[1], int(floor(stepsize)))

    fluxes = []
    rapix = []
    decpix = []
    signal = np.zeros(shape)
    xos, yos = np.meshgrid(xlocs,ylocs)

    print "making signal"
    xpix, ypix = np.meshgrid(range(shape[0]),range(shape[1]))
    for f, loc in enumerate(zip(np.ravel(xos),np.ravel(yos))):
        flux = 5 + f*2
        xo, yo = loc
        major, minor = smoothing, smoothing
        pa = 0
        src = ae.ntwodgaussian([flux, xo, yo, major, minor, pa])(xpix,ypix)
        fluxes.append(flux)
        rapix.append(xo)
        decpix.append(yo)
        signal += src

    print "source finding"

    hdulist[0].data = signal + noise
    sources = ae.find_sources_in_image(hdulist, max_summits=2, rms=1)
    # sub the new version in
    ae.multi_gauss = new_multi_gauss
    csources = ae.find_sources_in_image(hdulist, max_summits=2, rms=1)




    # convert all sources into something that I can crossmatch
    print "converting and xmatching"
    refsources = []
    major, minor = smoothing, smoothing
    pa = 0
    for f, xo, yo in zip(fluxes, rapix, decpix):
        ra, dec = ae.pix2sky([yo+1,xo+1]) # not sure why I need +1 here
        s = ae.SimpleSource() # careful to use ()!
        s.peak_flux = f
        s.ra = ra
        s.dec = dec
        s.a = major
        s.b = minor
        s.pa = pa
        refsources.append(s)

    # now do some crossmatching
    cpositions = SkyCoord([c.ra  for c in csources]*u.degree, [c.dec  for c in csources]*u.degree, frame='icrs')
    #print cpositions
    positions = SkyCoord([c.ra for c in sources]*u.degree, [c.dec for c in sources]*u.degree, frame='icrs')
    #print positions
    refpositions = SkyCoord([c.ra for c in refsources]*u.degree, [c.dec for c in refsources]*u.degree, frame='icrs')
    #print refpositions

    # uncorr noise
    index, dist2d, _ = refpositions.match_to_catalog_sky(positions)
    cindex, cdist2d, _ = refpositions.match_to_catalog_sky(cpositions)
    # print index, dist2d

    cflux, cflux_err, cflux_diff = [], [], []
    for i,j in enumerate(cindex):
        cflux.append(fluxes[i])
        cflux_err.append(csources[j].err_peak_flux)
        cflux_diff.append(abs(csources[j].peak_flux - refsources[i].peak_flux))

    flux, flux_err, flux_diff = [], [], []
    for i,j in enumerate(index):
        flux.append(fluxes[i])
        flux_err.append(sources[j].err_peak_flux)
        flux_diff.append(abs(sources[j].peak_flux - refsources[i].peak_flux))

    print "plotting"
    # plot the results
    fig = pyplot.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(flux, flux_err, 'bo-')
    ax1.plot(flux, flux_diff, 'ro-')
    ax1.set_title('uncorr errors')
    print "uncorr diff/err", ae.gmean(np.array(flux_diff)/np.array(flux_err))

    ax2 = fig.add_subplot(122)
    ax2.plot(cflux, cflux_err, 'bo-')
    ax2.plot(cflux, cflux_diff, 'ro-')
    ax2.set_title('corr errors')
    print "corr diff/err", ae.gmean(np.array(cflux_diff)/np.array(cflux_err))


    for a in [ax1,ax2]:
        a.set_xscale('log')
        a.set_yscale('log')
        a.legend(['reported','actual'],loc='lower left')
        a.set_ylim([1e-3,1e1])

    pyplot.show()

### NEW CODE THAT DOES WORK ###
def make_params(amp, xo, yo, sx, sy, theta, comp=0):
    prefix = "c{0}_".format(comp)
    params = lmfit.Parameters()
    params.add(prefix+'amp', value=amp)
    params.add(prefix+'xo', value=xo)
    params.add(prefix+'yo', value=yo)
    params.add(prefix+'sx', value=sx)
    params.add(prefix+'sy', value=sy)
    params.add(prefix+'theta', value=theta)
    return params


def gen_params():
    snrlist = np.logspace(np.log10(5),2,100,endpoint=True)
    palist = [0,15,30,45,60,75,90]
    majlist = np.linspace(1,5,10,endpoint=True)
    minlist = [1,2,3]
    xolist = []
    yolist = []

    for amp,xo,yo in zip(snrlist,xolist,yolist):
        yield make_params(amp,xo,yo,majlist[0],minlist[0],palist[0])


def gen_sources():
    for params in gen_params():
        #convert params to sky_params
        sky_params = params
        yield params, sky_params





def make_images():
    return


if __name__ == "__main__":
    #dk_error_comp()