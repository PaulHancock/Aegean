#! /usr/bin/env python
"""
Provide fitting routines and helper functions to Aegean
"""

import copy
import math
import lmfit
import numpy as np

from scipy.linalg import eigh, inv
from AegeanTools.logging import logger
from . import flags
from .angle_tools import bear, gcd
from .exceptions import AegeanNaNModelError

# don't freak out if numba isn't installed
try:
    from numba import njit
except ImportError:

    def njit(f):
        return f


__author__ = "Paul Hancock"

# ERR_MASK is used to indicate that the err_x value can't be determined
ERR_MASK = -1.0


# Modelling and fitting functions
@njit
def elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta):
    """
    Generate a model 2d Gaussian with the given parameters.
    Evaluate this model at the given locations x,y.

    Parameters
    ----------
    x, y : numeric or array-like
        locations at which to evaluate the gaussian
    amp : float
        Peak value.
    xo, yo : float
        Center of the gaussian.
    sx, sy : float
        major/minor axes in sigmas
    theta : float
        position angle (degrees) CCW from x-axis

    Returns
    -------
    data : numeric or array-like
        Gaussian function evaluated at the x,y locations.
    """
    if not np.isfinite(theta):
        sint = np.nan
        cost = np.nan
    else:
        sint = math.sin(np.radians(theta))
        cost = math.cos(np.radians(theta))

    xxo = x - xo
    yyo = y - yo
    exp = (xxo * cost + yyo * sint) ** 2 / sx**2 + (
        xxo * sint - yyo * cost
    ) ** 2 / sy**2
    exp *= -1.0 / 2
    return amp * np.exp(exp)


def elliptical_gaussian_with_alpha(
    x, y, v, amp, xo, yo, vo, sx, sy, theta, alpha, beta=None
):
    """
    Generate a model 2d Gaussian with spectral terms.
    Evaluate this model at the given locations x,y,dv.

    amp is the amplitude at the reference frequency vo

    The model is:
    S(x,v) = amp (v/vo) ^ (alpha + beta x log(v/vo))

    When beta is none it is ignored.

    Parameters
    ----------
    x, y, v : numeric or array-like
        locations at which to evaluate the gaussian
    amp : float
        Peak value.
    xo, yo, vo: float
        Center of the gaussian.
    sx, sy : float
        major/minor axes in sigmas
    theta : float
        position angle (degrees) CCW from x-axis

    alpha, beta: float
        The spectral terms of the fit.

    Returns
    -------
    data : numeric or array-like
        Gaussian function evaluated at the x,y locations.
    """
    exponent = alpha
    if beta is not None:
        exponent += beta * np.log10(v / vo)
    snu = amp * (v / vo) ** (exponent)
    gauss = elliptical_gaussian(x, y, snu, xo, yo, sx, sy, theta)
    return gauss


def Cmatrix(x, y, sx, sy, theta):
    """
    Construct a correlation matrix corresponding to the data.
    The matrix assumes a gaussian correlation function.

    Parameters
    ----------
    x, y : array-like
        locations at which to evaluate the correlation matrix
    sx, sy : float
        major/minor axes of the gaussian correlation function (sigmas)

    theta : float
        position angle of the gaussian correlation function (degrees)

    Returns
    -------
    data : array-like
        The C-matrix.
    """
    C = np.vstack(
        [elliptical_gaussian(x, y, 1, i, j, sx, sy, theta) for i, j in zip(x, y)]
    )
    return C


def Bmatrix(C):
    """
    Calculate a matrix which is effectively the square root of the
    correlation matrix C

    Parameters
    ----------
    C : 2d array
        A covariance matrix

    Returns
    -------
    B : 2d array
        A matrix B such the B.dot(B') = inv(C)
    """
    # this version of finding the square root of the inverse matrix
    # suggested by Cath Trott
    L, Q = eigh(C)
    # force very small eigenvalues to have some minimum non-zero value
    minL = 1e-9 * L[-1]
    L[L < minL] = minL
    S = np.diag(1 / np.sqrt(L))
    B = Q.dot(S)
    return B


def jacobian(pars, x, y):
    """
    Analytical calculation of the Jacobian for an elliptical gaussian
    Will work for a model that contains multiple Gaussians, and for which
    some components are not being fit (don't vary).

    Parameters
    ----------
    pars : lmfit.Model
        The model parameters
    x, y : list
        Locations at which the jacobian is being evaluated

    Returns
    -------
    j : 2d array
        The Jacobian.

    See Also
    --------
    :func:`AegeanTools.fitting.emp_jacobian`
    """

    matrix = []

    for i in range(int(pars["components"].value)):
        prefix = "c{0}_".format(i)
        amp = pars[prefix + "amp"].value
        xo = pars[prefix + "xo"].value
        yo = pars[prefix + "yo"].value
        sx = pars[prefix + "sx"].value
        sy = pars[prefix + "sy"].value
        theta = pars[prefix + "theta"].value

        # The derivative with respect to component i
        # doesn't depend on any other components thus
        # the model should not contain the other components
        model = elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta)

        # precompute for speed
        sint = np.sin(np.radians(theta))
        cost = np.cos(np.radians(theta))
        xxo = x - xo
        yyo = y - yo
        xcos, ycos = xxo * cost, yyo * cost
        xsin, ysin = xxo * sint, yyo * sint

        if pars[prefix + "amp"].vary:
            dmds = model / amp
            matrix.append(dmds)

        if pars[prefix + "xo"].vary:
            dmdxo = cost * (xcos + ysin) / sx**2 + sint * (xsin - ycos) / sy**2
            dmdxo *= model
            matrix.append(dmdxo)

        if pars[prefix + "yo"].vary:
            dmdyo = sint * (xcos + ysin) / sx**2 - cost * (xsin - ycos) / sy**2
            dmdyo *= model
            matrix.append(dmdyo)

        if pars[prefix + "sx"].vary:
            dmdsx = model / sx**3 * (xcos + ysin) ** 2
            matrix.append(dmdsx)

        if pars[prefix + "sy"].vary:
            dmdsy = model / sy**3 * (xsin - ycos) ** 2
            matrix.append(dmdsy)

        if pars[prefix + "theta"].vary:
            dmdtheta = (
                model * (sy**2 - sx**2) * (xsin - ycos) * (xcos + ysin) / sx**2 / sy**2
            )
            matrix.append(dmdtheta)

    return np.array(matrix)


def emp_jacobian(pars, x, y):
    """
    An empirical calculation of the Jacobian
    Will work for a model that contains multiple Gaussians, and for which
    some components are not being fit (don't vary).

    Parameters
    ----------
    pars : lmfit.Model
        The model parameters
    x, y : list
        Locations at which the jacobian is being evaluated

    Returns
    -------
    j : 2d array
        The Jacobian.

    See Also
    --------
    :func:`AegeanTools.fitting.jacobian`
    """
    eps = 1e-5
    matrix = []
    model = ntwodgaussian_lmfit(pars)(x, y)
    for i in range(int(pars["components"].value)):
        prefix = "c{0}_".format(i)
        for p in ["amp", "xo", "yo", "sx", "sy", "theta"]:
            if pars[prefix + p].vary:
                pars[prefix + p].value += eps
                dmdp = ntwodgaussian_lmfit(pars)(x, y) - model
                matrix.append(dmdp / eps)
                pars[prefix + p].value -= eps
    matrix = np.array(matrix)
    return matrix


def lmfit_jacobian(pars, x, y, errs=None, B=None, emp=False):
    r"""
    Wrapper around `AegeanTools.fitting.jacobian` and
    `AegeanTools.fitting.emp_jacobian` which gives the output in a format
    that is required for lmfit.

    Parameters
    ----------
    pars : lmfit.Model
        The model parameters

    x, y : list
        Locations at which the jacobian is being evaluated

    errs : list
        a vector of 1\sigma errors (optional). Default = None

    B : 2d-array
        a B-matrix (optional) see `AegeanTools.fitting.Bmatrix`

    emp : bool
        If true the use empirical Jacobian, otherwise use analytical Default =
        False.

    Returns
    -------
    j : 2d-array
        A Jacobian.

    See Also
    --------
    `AegeanTools.fitting.Bmatrix`
    `AegeanTools.fitting.jacobian`
    `AegeanTools.fitting.emp_jacobian`

    """
    if emp:
        matrix = emp_jacobian(pars, x, y)
    else:
        # calculate in the normal way
        matrix = jacobian(pars, x, y)
    # now munge this to be as expected for lmfit
    matrix = np.vstack(matrix)

    if errs is not None:
        matrix /= errs
        # matrix = matrix.dot(errs)

    if B is not None:
        matrix = matrix.dot(B)

    matrix = np.transpose(matrix)
    return matrix


def errors(source, model, wcshelper):
    """
    Convert pixel based errors into sky coord errors

    Parameters
    ----------
    source : :class:`AegeanTools.models.SimpleSource`
        The source which was fit.

    model : lmfit.Parameters
        The model which was fit.

    wcshelper : :class:`AegeanTools.wcs_helpers.WCSHelper`
        WCS information.

    Returns
    -------
    source : :class:`AegeanTools.models.SimpleSource`
        The modified source obejct.

    """

    # if the source wasn't fit then all errors are -1
    if source.flags & (flags.NOTFIT | flags.FITERR):
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = (
            source.err_ra
        ) = source.err_dec = source.err_int_flux = ERR_MASK
        return source
    # copy the errors from the model
    prefix = "c{0}_".format(source.source)
    err_amp = model[prefix + "amp"].stderr
    xo, yo = model[prefix + "xo"].value, model[prefix + "yo"].value
    err_xo = model[prefix + "xo"].stderr
    err_yo = model[prefix + "yo"].stderr

    sx, sy = model[prefix + "sx"].value, model[prefix + "sy"].value
    err_sx = model[prefix + "sx"].stderr
    err_sy = model[prefix + "sy"].stderr

    theta = model[prefix + "theta"].value
    err_theta = model[prefix + "theta"].stderr

    source.err_peak_flux = err_amp
    pix_errs = [err_xo, err_yo, err_sx, err_sy, err_theta]

    logger.debug("Pix errs: {0}".format(pix_errs))

    ref = wcshelper.pix2sky([xo, yo])
    # check to see if the reference position has a valid WCS coordinate
    # It is possible for this to fail,
    # even if the ra/dec conversion works elsewhere
    if not all(np.isfinite(ref)):
        source.flags |= flags.WCSERR
        source.err_peak_flux = source.err_a = source.err_b = ERR_MASK
        source.err_pa = source.err_ra = source.err_dec = ERR_MASK
        source.err_int_flux = ERR_MASK
        return source

    # position errors
    if (
        model[prefix + "xo"].vary
        and model[prefix + "yo"].vary
        and all(np.isfinite([err_xo, err_yo]))
    ):
        offset = wcshelper.pix2sky([xo + err_xo, yo + err_yo])
        source.err_ra = gcd(ref[0], ref[1], offset[0], ref[1])
        source.err_dec = gcd(ref[0], ref[1], ref[0], offset[1])
    else:
        source.err_ra = source.err_dec = -1

    if model[prefix + "theta"].vary and np.isfinite(err_theta):
        # pa error
        off1 = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))]
        )
        off2 = wcshelper.pix2sky(
            [
                xo + sx * np.cos(np.radians(theta + err_theta)),
                yo + sy * np.sin(np.radians(theta + err_theta)),
            ]
        )
        source.err_pa = abs(
            bear(ref[0], ref[1], off1[0], off1[1])
            - bear(ref[0], ref[1], off2[0], off2[1])
        )
    else:
        source.err_pa = ERR_MASK

    if (
        model[prefix + "sx"].vary
        and model[prefix + "sy"].vary
        and all(np.isfinite([err_sx, err_sy]))
    ):
        # major axis error
        ref = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))]
        )
        offset = wcshelper.pix2sky(
            [
                xo + (sx + err_sx) * np.cos(np.radians(theta)),
                yo + sy * np.sin(np.radians(theta)),
            ]
        )
        source.err_a = gcd(ref[0], ref[1], offset[0], offset[1]) * 3600

        # minor axis error
        ref = wcshelper.pix2sky(
            [
                xo + sx * np.cos(np.radians(theta + 90)),
                yo + sy * np.sin(np.radians(theta + 90)),
            ]
        )
        offset = wcshelper.pix2sky(
            [
                xo + sx * np.cos(np.radians(theta + 90)),
                yo + (sy + err_sy) * np.sin(np.radians(theta + 90)),
            ]
        )
        source.err_b = gcd(ref[0], ref[1], offset[0], offset[1]) * 3600
    else:
        source.err_a = source.err_b = ERR_MASK

    sqerr = 0
    sqerr += (
        (source.err_peak_flux / source.peak_flux) ** 2
        if source.err_peak_flux > 0
        else 0
    )
    sqerr += (source.err_a / source.a) ** 2 if source.err_a > 0 else 0
    sqerr += (source.err_b / source.b) ** 2 if source.err_b > 0 else 0
    if sqerr == 0:
        source.err_int_flux = ERR_MASK
    else:
        source.err_int_flux = abs(source.int_flux * np.sqrt(sqerr))

    return source


def new_errors(source, model, wcshelper):  # pragma: no cover
    """
    Convert pixel based errors into sky coord errors
    Uses covariance matrix for ra/dec errors
    and calculus approach to a/b/pa errors

    Parameters
    ----------
    source : :class:`AegeanTools.models.SimpleSource`
        The source which was fit.

    model : lmfit.Parameters
        The model which was fit.

    wcshelper : :class:`AegeanTools.wcs_helpers.WCSHelper`
        WCS information.

    Returns
    -------
    source : :class:`AegeanTools.models.SimpleSource`
        The modified source obejct.

    """

    # if the source wasn't fit then all errors are -1
    if source.flags & (flags.NOTFIT | flags.FITERR):
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = (
            source.err_ra
        ) = source.err_dec = source.err_int_flux = ERR_MASK
        return source
    # copy the errors/values from the model
    prefix = "c{0}_".format(source.source)
    err_amp = model[prefix + "amp"].stderr
    xo, yo = model[prefix + "xo"].value, model[prefix + "yo"].value
    err_xo = model[prefix + "xo"].stderr
    err_yo = model[prefix + "yo"].stderr

    sx, sy = model[prefix + "sx"].value, model[prefix + "sy"].value
    err_sx = model[prefix + "sx"].stderr
    err_sy = model[prefix + "sy"].stderr

    theta = model[prefix + "theta"].value
    err_theta = model[prefix + "theta"].stderr

    # the peak flux error doesn't need to be converted, just copied
    source.err_peak_flux = err_amp

    pix_errs = [err_xo, err_yo, err_sx, err_sy, err_theta]

    # check for inf/nan errors -> these sources have poor fits.
    if not all(a is not None and np.isfinite(a) for a in pix_errs):
        source.flags |= flags.FITERR
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = (
            source.err_ra
        ) = source.err_dec = source.err_int_flux = ERR_MASK
        return source

    # calculate the reference coordinate
    ref = wcshelper.pix2sky([xo, yo])
    # check to see if the reference position has a valid WCS coordinate
    # It is possible for this to fail,
    # even if the ra/dec conversion works elsewhere
    if not all(np.isfinite(ref)):
        source.flags |= flags.WCSERR
        source.err_peak_flux = source.err_a = source.err_b = source.err_pa = (
            source.err_ra
        ) = source.err_dec = source.err_int_flux = ERR_MASK
        return source

    # calculate position errors by transforming the error ellipse
    if model[prefix + "xo"].vary and model[prefix + "yo"].vary:
        # determine the error ellipse from the Jacobian
        mat = model.covar[1:3, 1:3]
        if not (np.all(np.isfinite(mat))):
            source.err_ra = source.err_dec = ERR_MASK
        else:
            (a, b), e = np.linalg.eig(mat)
            pa = np.degrees(np.arctan2(*e[0]))
            # transform this ellipse into sky coordinates
            _, _, major, minor, pa = wcshelper.pix2sky_ellipse([xo, yo], a, b, pa)

            # determine the radius of the ellipse along the ra/dec directions.
            source.err_ra = (
                major
                * minor
                / np.hypot(
                    major * np.sin(np.radians(pa)), minor * np.cos(np.radians(pa))
                )
            )
            source.err_dec = (
                major
                * minor
                / np.hypot(
                    major * np.cos(np.radians(pa)), minor * np.sin(np.radians(pa))
                )
            )
    else:
        source.err_ra = source.err_dec = -1

    if model[prefix + "theta"].vary:
        # pa error
        off1 = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))]
        )
        # offset by 1 degree
        off2 = wcshelper.pix2sky(
            [
                xo + sx * np.cos(np.radians(theta + 1)),
                yo + sy * np.sin(np.radians(theta + 1)),
            ]
        )
        # scale the initial theta error by this amount
        source.err_pa = (
            abs(
                bear(ref[0], ref[1], off1[0], off1[1])
                - bear(ref[0], ref[1], off2[0], off2[1])
            )
            * err_theta
        )
    else:
        source.err_pa = ERR_MASK

    if model[prefix + "sx"].vary and model[prefix + "sy"].vary:
        # major axis error
        ref = wcshelper.pix2sky(
            [xo + sx * np.cos(np.radians(theta)), yo + sy * np.sin(np.radians(theta))]
        )
        # offset by 0.1 pixels
        offset = wcshelper.pix2sky(
            [
                xo + (sx + 0.1) * np.cos(np.radians(theta)),
                yo + sy * np.sin(np.radians(theta)),
            ]
        )
        source.err_a = gcd(ref[0], ref[1], offset[0], offset[1]) / 0.1 * err_sx * 3600

        # minor axis error
        ref = wcshelper.pix2sky(
            [
                xo + sx * np.cos(np.radians(theta + 90)),
                yo + sy * np.sin(np.radians(theta + 90)),
            ]
        )
        # offset by 0.1 pixels
        offset = wcshelper.pix2sky(
            [
                xo + sx * np.cos(np.radians(theta + 90)),
                yo + (sy + 0.1) * np.sin(np.radians(theta + 90)),
            ]
        )
        source.err_b = gcd(ref[0], ref[1], offset[0], offset[1]) / 0.1 * err_sy * 3600
    else:
        source.err_a = source.err_b = ERR_MASK
    sqerr = 0
    sqerr += (
        (source.err_peak_flux / source.peak_flux) ** 2
        if source.err_peak_flux > 0
        else 0
    )
    sqerr += (source.err_a / source.a) ** 2 if source.err_a > 0 else 0
    sqerr += (source.err_b / source.b) ** 2 if source.err_b > 0 else 0
    source.err_int_flux = abs(source.int_flux * np.sqrt(sqerr))

    return source


def ntwodgaussian_lmfit(params):
    """
    Convert an lmfit.Parameters object into a function which calculates the
    model.


    Parameters
    ----------
    params : lmfit.Parameters
        Model parameters, can have multiple components.

    Returns
    -------
    model : func
        A function f(x,y) that will compute the model.
    """

    def rfunc(x, y):
        """
        Compute the model given by params, at pixel coordinates x,y

        Parameters
        ----------
        x, y : numpy.ndarray
            The x/y pixel coordinates at which the model is being evaluated

        Returns
        -------
        result : numpy.ndarray
            Model
        """
        result = None
        for i in range(int(params["components"].value)):
            prefix = "c{0}_".format(i)
            # I hope this doesn't kill our run time
            amp = np.nan_to_num(params[prefix + "amp"].value)
            xo = params[prefix + "xo"].value
            yo = params[prefix + "yo"].value
            sx = params[prefix + "sx"].value
            sy = params[prefix + "sy"].value
            theta = params[prefix + "theta"].value
            if result is not None:
                result += elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta)
            else:
                result = elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta)
        return result

    return rfunc


def nthreedgaussian_lmfit(params):
    """
    Convert an lmfit.Parameters object into a function which calculates the
    model.


    Parameters
    ----------
    params : lmfit.Parameters
        Model parameters, can have multiple components.

    Returns
    -------
    model : func
        A function f(x,y) that will compute the model.
    """

    def rfunc(v, x, y):  # TODO: Update the Doc string
        #! v is not a pixel coordinate it should be actual frequency i.e. pix2freq
        """
        Compute the model given by params, at pixel coordinates x,y

        Parameters
        ----------
        x, y : numpy.ndarray
            The x/y pixel coordinates at which the model is being evaluated

        Returns
        -------
        result : numpy.ndarray
            Model
        """
        result = None

        for i in range(int(params["components"].value)):
            prefix = f"c{i}_"
            # I hope this doesn't kill our run time
            amp = np.nan_to_num(params[prefix + "amp"].value)
            xo = params[prefix + "xo"].value
            yo = params[prefix + "yo"].value
            sx = params[prefix + "sx"].value
            sy = params[prefix + "sy"].value
            theta = params[prefix + "theta"].value
            alpha = params[prefix + "alpha"].value
            nu0 = params[prefix + "nu0"].value
            if result is not None:
                result += elliptical_gaussian_with_alpha(
                    x, y, v, amp, xo, yo, nu0, sx, sy, theta, alpha
                )  # TODO: Pass the frequency into this, which is the current frequency
            else:
                result = elliptical_gaussian_with_alpha(
                    x, y, v, amp, xo, yo, nu0, sx, sy, theta, alpha
                )
        return result

    return rfunc


def do_lmfit(data, params, B=None, errs=None, dojac=False):
    """
    Fit the model to the data
    data may contain 'flagged' or 'masked' data with the value of np.nan

    Parameters
    ----------
    data : 2d-array
        Image data

    params : lmfit.Parameters
        Initial model guess.

    B : 2d-array
        B matrix to be used in residual calculations.
        Default = None.

    errs : 1d-array

    dojac : bool
        If true then an analytic jacobian will be passed to the fitter

    Returns
    -------
    result : ?
        lmfit.minimize result.

    params : lmfit.Params
        Fitted model.

    See Also
    --------
    :func:`AegeanTools.fitting.lmfit_jacobian`

    """
    # copy the params so as not to change the initial conditions
    # in case we want to use them elsewhere
    params = copy.deepcopy(params)
    data = np.array(data)
    mask = np.where(np.isfinite(data))

    def residual(params, x, y, B=None, errs=None):
        """
        The residual function required by lmfit

        Parameters
        ----------
        params: lmfit.Params
            The parameters of the model being fit

        Returns
        -------
        result : numpy.ndarray
            Model - Data
        """
        f = ntwodgaussian_lmfit(params)  # A function describing the model
        model = f(*mask)  # The actual model

        if np.any(~np.isfinite(model)):
            raise AegeanNaNModelError(
                "lmfit optimisation has return NaN in the parameter set. "
            )

        if B is None:
            return model - data[mask]
        else:
            return (model - data[mask]).dot(B)

    if dojac:
        result = lmfit.minimize(
            residual,
            params,
            kws={"x": mask[0], "y": mask[1], "B": B, "errs": errs},
            Dfun=lmfit_jacobian,
        )
    else:
        result = lmfit.minimize(
            residual, params, kws={"x": mask[0], "y": mask[1], "B": B, "errs": errs}
        )

    # Remake the residual so that it is once again (model - data)
    if B is not None:
        result.residual = result.residual.dot(inv(B))
    return result, params


def do_lmfit_3D(
    data,
    params,
    freq_mapping=None,
    B=None,
    errs=None,
    dojac=False,
):  # TODO: Go through B matrix
    """
    Fit the model to the data
    data may contain 'flagged' or 'masked' data with the value of np.NaN

    Parameters
    ----------
    data : 2d-array
        Image data

    params : lmfit.Parameters
        Initial model guess.

    B : 2d-array
        B matrix to be used in residual calculations.
        Default = None.

    errs : 1d-array

    dojac : bool
        If true then an analytic jacobian will be passed to the fitter

    Returns
    -------
    result : ?
        lmfit.minimize result.

    params : lmfit.Params
        Fitted model.

    See Also
    --------
    :func:`AegeanTools.fitting.lmfit_jacobian`

    """
    # copy the params so as not to change the initial conditions
    # in case we want to use them elsewhere
    params = copy.deepcopy(params)
    data = np.array(data)
    mask = np.where(np.isfinite(data))
    fmask = freq_mapping[mask[0]]

    def residual(params, x, y, B=None, errs=None):
        """
        The residual function required by lmfit

        Parameters
        ----------
        params: lmfit.Params
            The parameters of the model being fit

        Returns
        -------
        result : numpy.ndarray
            Model - Data
        """
        f = nthreedgaussian_lmfit(params)  # A function describing the model
        model = f(fmask, mask[1], mask[2])  # The actual model

        if np.any(~np.isfinite(model)):
            logger.debug(f"The parameters are  {params}")
            raise AegeanNaNModelError(
                "lmfit optimisation has return NaN in the parameter set. "
            )

        if B is None:
            return model - data[mask]
        else:
            return (model - data[mask]).dot(B)

    # if dojac:
    #     result = lmfit.minimize(
    #         residual,
    #         params,
    #         kws={"x": mask[0], "y": mask[1], "B": B, "errs": errs},
    #         Dfun=lmfit_jacobian,
    #    )
    result = lmfit.minimize(
        residual, params, kws={"x": mask[0], "y": mask[1], "B": B, "errs": errs}
    )

    # Remake the residual so that it is once again (model - data)
    # if B is not None:
    #     result.residual = result.residual.dot(inv(B))
    return result, params


def covar_errors(params, data, errs, B, C=None):
    r"""
    Take a set of parameters that were fit with lmfit, and replace the errors
    with the 1\sigma errors calculated using the covariance matrix.


    Parameters
    ----------
    params : lmfit.Parameters
        Model

    data : 2d-array
        Image data

    errs : 2d-array ?
        Image noise.

    B : 2d-array
        B matrix.

    C : 2d-array
        C matrix. Optional. If supplied then Bmatrix will not be used.

    Returns
    -------
    params : lmfit.Parameters
        Modified model.
    """

    mask = np.where(np.isfinite(data))

    # calculate the proper parameter errors and copy them across.
    if C is not None:
        try:
            J = lmfit_jacobian(params, mask[0], mask[1], errs=errs)
            covar = np.transpose(J).dot(inv(C)).dot(J)
            onesigma = np.sqrt(np.diag(inv(covar)))
        except (np.linalg.linalg.LinAlgError, ValueError):
            C = None

    if C is None:
        try:
            J = lmfit_jacobian(params, mask[0], mask[1], B=B, errs=errs)
            covar = np.transpose(J).dot(J)
            onesigma = np.sqrt(np.diag(inv(covar)))
        except (np.linalg.linalg.LinAlgError, ValueError):
            onesigma = [-2] * len(mask[0])

    for i in range(int(params["components"].value)):
        prefix = "c{0}_".format(i)
        j = 0
        for p in ["amp", "xo", "yo", "sx", "sy", "theta"]:
            if params[prefix + p].vary:
                params[prefix + p].stderr = onesigma[j]
                j += 1

    return params
