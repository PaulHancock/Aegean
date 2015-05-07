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
    sint, cost = math.sin(theta), math.cos(theta)
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


def emp_jacobian(pars, x, y, errs=None):
    """
    An empirical calculation of the jacobian
    :param pars:
    :param x:
    :param y:
    :return:
    """
    eps=1e-5
    matrix = []
    model = ntwodgaussian_lmfit(pars)
    for i in xrange(pars.components):
        prefix = "c{0}_".format(i)
        # Note: all derivatives are calculated, even if the parameter is fixed
        for p in ['amp','xo','yo','sx','sy','theta']:
            pars[prefix+p].value += eps
            dmdp = ntwodgaussian_lmfit(pars) - model
            matrix.append(dmdp)
            pars[prefix+p].value -= eps

    matrix = np.array(matrix)/eps
    if errs is not None:
        matrix /=errs**2
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


def ntwodgaussian_lmfit(params):
    """
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
                result += elliptical_gaussian(x,y,amp,xo,yo,sx,sy,np.radians(theta))
            else:
                result =  elliptical_gaussian(x,y,amp,xo,yo,sx,sy,np.radians(theta))
        return result
    return rfunc


def do_lmfit(data, params):
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

    B = None
    #mx, my = mask
    #pixbeam = get_pixbeam()
    #C = Cmatrix(mx, my, pixbeam.a*fwhm2cc, pixbeam.b*fwhm2cc, pixbeam.pa)
    #B = Bmatrix(C)

    def residual(params):
        f = ntwodgaussian_lmfit(params)  # A function describing the model
        model = f(*mask)  # The actual model
        if B is None:
            return model-data[mask]
        else:
            return (model - data[mask]).dot(B)

    result = lmfit.minimize(residual, params)
    return result, params

