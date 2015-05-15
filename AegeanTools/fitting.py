#! /usr/bin/env python

"""
Provide fitting routines and helper fucntions to Aegean
"""

__author__ = "Paul Hancock"

import sys
import copy
import math
import numpy as np
from scipy.linalg import eigh, inv
import lmfit

# Other AegeanTools
# from models import OutputSource, IslandSource


# join the Aegean logger
import logging
log = logging.getLogger('Aegean')

# Modelling and fitting functions
def elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta):
    """
    Generate a model 2d Gaussian with the given parameters.
    Evaluate this model at the given locations x,y.

    :param x,y: locations at which to calculate values
    :param amp: amplitude of Gaussian
    :param xo,yo: position of Gaussian
    :param major,minor: axes (sigmas)
    :param theta: position angle (radians) CCW from x-axis
    :return: Gaussian function evaluated at x,y locations
    """
    sint, cost = math.sin(np.radians(theta)), math.cos(np.radians(theta))
    xxo = x-xo
    yyo = y-yo
    exp = (xxo*cost + yyo*sint)**2 / sx**2 \
        + (xxo*sint - yyo*cost)**2 / sy**2
    exp *=-1./2
    return amp*np.exp(exp)


def Cmatrix(x,y,sx,sy,theta):
    """
    Construct a correlation matrix corresponding to the data.
    :param x:
    :param y:
    :param sx:
    :param sy:
    :param theta:
    :return:
    """
    f = lambda i,j: elliptical_gaussian(x,y,1,i,j,sx,sy,theta)
    C = np.vstack( [ f(i,j) for i,j in zip(x,y)] )
    return C


def Bmatrix(C):
    """
    Calculate a matrix which is effectively the square root of the correlation matrix C
    :param C:
    :return: A matrix B such the B.dot(B') = inv(C)
    """
    # this version of finding the square root of the inverse matrix
    # suggested by Cath,
    L,Q = eigh(C)
    if not all(L>0):
        log.critical("at least one eigenvalue is negative, this will cause problems!")
        sys.exit(1)
    S = np.diag(1/np.sqrt(L))
    B = Q.dot(S)
    return B


def emp_jacobian(pars, x, y, errs=None, B=None):
    """
    An empirical calculation of the jacobian
    :param pars:
    :param x:
    :param y:
    :return:
    """
    eps=1e-5
    matrix = []
    model = ntwodgaussian_lmfit(pars)(x,y)
    for i in xrange(pars.components):
        prefix = "c{0}_".format(i)
        for p in ['amp','xo','yo','sx','sy','theta']:
            if pars[prefix+p].vary:
                pars[prefix+p].value += eps
                dmdp = ntwodgaussian_lmfit(pars)(x,y) - model
                matrix.append(dmdp/eps)
                pars[prefix+p].value -= eps
    matrix = np.array(matrix)
    if errs is not None:
        matrix /=errs

    if B is not None:
        matrix = matrix.dot(B)
    matrix = np.transpose(matrix)
    return matrix


def jacobian(pars, x, y, errs=None, B=None):

    matrix = []
    model = ntwodgaussian_lmfit(pars)(x,y)

    for i in xrange(pars.components):
        prefix = "c{0}_".format(i)
        amp = pars[prefix+'amp'].value
        xo = pars[prefix+'xo'].value
        yo = pars[prefix+'yo'].value
        sx = pars[prefix+'sx'].value
        sy = pars[prefix+'sy'].value
        theta  = pars[prefix+'theta'].value

        # precompute for speed
        sint = np.sin(np.radians(theta))
        cost = np.cos(np.radians(theta))
        xxo = x-xo
        yyo = y-yo
        xcos, ycos = xxo*cost, yyo*cost
        xsin, ysin = xxo*sint, yyo*sint

        if pars[prefix+'amp'].vary:
            dmds = model/amp
            matrix.append(dmds)

        if pars[prefix+'xo'].vary:
            dmdxo = cost * (xcos + ysin) /sx**2 + sint * (xsin - ycos) /sy**2
            dmdxo *= model
            matrix.append(dmdxo)

        if pars[prefix+'yo'].vary:
            dmdyo = sint * (xcos + ysin) /sx**2 - cost * (xsin - ycos) /sy**2
            dmdyo *= model
            matrix.append(dmdyo)

        if pars[prefix+'sx'].vary:
            dmdsx = model / sx**3 * (xcos + ysin)**2
            matrix.append(dmdsx)

        if pars[prefix+'sy'].vary:
            dmdsy = model / sy**3 * (xsin - ycos)**2
            matrix.append(dmdsy)

        if pars[prefix+'theta'].vary:
            dmdtheta = model * (sx**2 - sy**2) * (xsin + ycos) * (xcos + ysin) / sx**2/sy**2
            matrix.append(dmdtheta)

    matrix=np.vstack(matrix)

    if errs is not None:
        matrix /=errs

    if B is not None:
        matrix = matrix.dot(B)

    matrix = np.transpose(matrix)
    return matrix


def CRB_errs(jac, C, B=None):
    """
    Calculate minimum errors given by the Cramer-Rao bound
    :param jac: the jacobian
    :param C: the correlation matrix
    :param B: B.dot(B') should = inv(C), ie B ~ sqrt(inv(C))
    :return: array of errors for the model parameters
    """
    if B is not None:
        fim_inv =  inv(np.transpose(jac).dot(B).dot(np.transpose(B)).dot(jac))
    else:
        fim = np.transpose(jac).dot(inv(C)).dot(jac)
        fim_inv = inv(fim)
    errs = np.sqrt(np.diag(fim_inv))
    return errs


def calc_errors(source, thetaN=None):
    """
    Calculate the parameter errors for a fitted source
    using the description of Condon'97
    All parameters are assigned errors, assuming that all params were fit.
    If some params were held fixed then these errors are overestimated.
    :param source: Source for which errors need to be calculated
    :return: The same source but with errors assigned.
    """

    # indices for the calculation or rho
    alphas = {'amp':(3./2, 3./2),
              'major':(5./2, 1./2),
              'xo':(5./2, 1./2),
              'minor':(1./2, 5./2),
              'yo':(1./2, 5./2),
              'pa':(1./2, 5./2)}

    major = source.a/3600 # degrees
    minor = source.b/3600 # degrees
    phi = np.radians(source.pa)
    if thetaN is None:
        log.critical(" you need to supply thetaN")
        thetaN = np.sqrt(get_beamarea_deg2(source.ra,source.dec)/np.pi)
    smoothing = major*minor / (thetaN**2)
    factor1 = (1 + (major / thetaN))
    factor2 = (1 + (minor / thetaN))
    snr = source.peak_flux/source.local_rms
    # calculation of rho2 depends on the parameter being used so we lambda this into a function
    rho2 = lambda x: smoothing/4 *factor1**alphas[x][0] * factor2**alphas[x][1] *snr**2

    source.err_peak_flux = source.peak_flux * np.sqrt(2/rho2('amp'))
    source.err_a = major * np.sqrt(2/rho2('major')) *3600 # arcsec
    source.err_b = minor * np.sqrt(2/rho2('minor')) *3600 # arcsec

    # TODO: proper conversion of x/y errors in ra/dec errors
    err_xo2 = 2./rho2('xo')*major**2/(8*np.log(2)) # Condon'97 eq 21
    err_yo2 = 2./rho2('yo')*minor**2/(8*np.log(2))
    source.err_ra  = np.sqrt( err_xo2*np.sin(phi)**2 + err_yo2*np.cos(phi)**2)
    source.err_dec = np.sqrt( err_xo2*np.cos(phi)**2 + err_yo2*np.sin(phi)**2)

    # if major/minor are very similar then we should not be able to figure out what pa is.
    if abs((major/minor)**2+(minor/major)**2 -2) < 0.01:
        source.err_pa = -1
    else:
        source.err_pa = np.degrees(np.sqrt(4/rho2('pa')) * (major*minor/(major**2-minor**2)))

    # integrated flux error
    err2 = (source.err_peak_flux/source.peak_flux)**2
    err2 += (thetaN**2/(major*minor)) *( (source.err_a/source.a)**2 + (source.err_b/source.b)**2)
    source.err_int_flux =source.int_flux * np.sqrt(err2)
    return


def ntwodgaussian_lmfit(params):
    """
    theta is in degrees
    :param params: model parameters (can be multiple)
    :return: a functiont that maps (x,y) -> model
    """
    def rfunc(x, y):
        result=None
        for i in range(params.components):
            prefix = "c{0}_".format(i)
            amp = params[prefix+'amp'].value
            xo = params[prefix+'xo'].value
            yo = params[prefix+'yo'].value
            sx = params[prefix+'sx'].value
            sy = params[prefix+'sy'].value
            theta = params[prefix+'theta'].value
            if result is not None:
                result += elliptical_gaussian(x,y,amp,xo,yo,sx,sy,theta)
            else:
                result =  elliptical_gaussian(x,y,amp,xo,yo,sx,sy,theta)
        return result
    return rfunc


def do_lmfit(data, params, B=None, errs=None):
    """
    Fit the model to the data
    data may contain 'flagged' or 'masked' data with the value of np.NaN
    input: data - pixel information
           params - and lmfit.Model instance
    return: fit results, modified model
    """
    # copy the params so as not to change the initial conditions
    # in case we want to use them elsewhere
    params = copy.deepcopy(params)
    data = np.array(data)
    mask = np.where(np.isfinite(data))

    def residual(params, B, **kwargs):
        f = ntwodgaussian_lmfit(params)  # A function describing the model
        model = f(*mask)  # The actual model
        if B is None:
            return model-data[mask]
        else:
            return (model - data[mask]).dot(B)

    result = lmfit.minimize(residual, params, kws={'x':mask[0],'y':mask[1],'B':B,'errs':errs}, Dfun = jacobian)

    return result, params


def covar_errors(params, data, errs, B):

    mask = np.where(np.isfinite(data))
    # errs = []
    # # Calculate the SNR for each source:
    # for i in xrange(params.components):
    #     prefix = "c{0}_".format(i)
    #     x = int(math.floor(params[prefix+'xo']))
    #     y = int(math.floor(params[prefix+'yo']))
    #     free_vars = len( [ 1 for a in params.keys() if params[a].vary and prefix in a])
    #     errs.extend( [rms[x,y]]*free_vars)
    #
    # #errs = np.vstack( errs,len(mask[0]))
    # #print errs
    # errs = errs[0]

    # now calculate the proper parameter errors and copy them across.
    #J = emp_jacobian(params, mask[0], mask[1], B=B, errs=errs)
    J = jacobian(params, mask[0], mask[1], B=B, errs=errs)
    covar = np.transpose(J).dot(J)
    onesigma = np.sqrt(np.diag(inv(covar)))
    for i in xrange(params.components):
        prefix = "c{0}_".format(i)
        j=0
        for p in ['amp','xo','yo','sx','sy','theta']:
            if params[prefix+p].vary:
                params[prefix+p].stderr = onesigma[j]
                j+=1

    return params


def ntwodgaussian_mpfit(inpars):
    """
    Return an array of values represented by multiple Gaussians as parametrized
    by params = [amp,x0,y0,major,minor,pa]{n}
    x0,y0,major,minor are in pixels
    major/minor are interpreted as being sigmas not FWHMs
    pa is in degrees
    """
    try:
        params = np.array(inpars).reshape(len(inpars) / 6, 6)
    except ValueError as e:
        if 'size' in e.message:
            log.error("inpars requires a multiple of 6 parameters")
            log.error("only {0} parameters supplied".format(len(inpars)))
        raise e

    def rfunc(x, y):
        result = None
        for p in params:
            amp, xo, yo, major, minor, pa = p
            if result is not None:
                result += elliptical_gaussian(x,y,amp,xo,yo,major,minor,pa)
            else:
                result =  elliptical_gaussian(x,y,amp,xo,yo,major,minor,pa)
        return result

    return rfunc


def test_jacobian():
    nx = 15
    ny = 12
    x,y = np.where(np.ones((nx,ny))==1)

    #smoothing = 1.27 # 3pix/beam
    #smoothing = 2.12 # 5pix/beam
    smoothing = 1.5 # ~4.2pix/beam

    # The model parameters
    params = lmfit.Parameters()
    params.add('c0_amp', value=1, min=0.5, max=2)
    params.add('c0_xo', value=1.*nx/2, min=nx/2.-smoothing/2., max=nx/2.+smoothing/2)
    params.add('c0_yo', value=1.*ny/2, min=ny/2.-smoothing/2., max=ny/2.+smoothing/2.)
    params.add('c0_sx', value=2*smoothing, min=0.8*smoothing)
    params.add('c0_sy', value=smoothing, min=0.8*smoothing)
    params.add('c0_theta',value=45)#, min=-2*np.pi, max=2*np.pi)
    params.components=1

    def rmlabels(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    from matplotlib import pyplot
    fig=pyplot.figure(1)
    # This sets all nan pixels to be a nasty yellow colour
    cmap = pyplot.cm.cubehelix
    cmap.set_bad('y',1.)
    #kwargs = {'interpolation':'nearest','cmap':cmap,'vmin':-0.1,'vmax':1, 'origin':'lower'}
    kwargs = {'interpolation':'nearest','cmap':cmap, 'origin':'lower'}
    for i,jac in enumerate([emp_jacobian,jacobian]):
        fig = pyplot.figure(i+1,figsize=(4,6))
        jdata = jac(params, x, y)
        fig.suptitle(str(jac))
        for k,p in enumerate(['amp','xo','yo','sx','sy','theta']):
            ax = fig.add_subplot(3,2,k+1)
            ax.imshow(jdata[:,k].reshape(nx,ny),**kwargs)
            ax.set_title(p)
            rmlabels(ax)

    pyplot.show()


if __name__ == "__main__":
    test_jacobian()