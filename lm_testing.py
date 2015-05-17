#! /usr/bin/env python

import numpy as np
import lmfit
import math
import sys
from AegeanTools.mpfit import mpfit
import logging
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.linalg import sqrtm, eigh, inv, pinv
import copy

from AegeanTools.fitting import elliptical_gaussian, Cmatrix, Bmatrix, emp_jacobian, CRB_errs, do_lmfit, ntwodgaussian_lmfit
from AegeanTools.fitting import jacobian as ana_jacobian

def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5*((x-cen)/sigma)**2)


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
            logging.error("inpars requires a multiple of 6 parameters")
            logging.error("only {0} parameters supplied".format(len(inpars)))
        raise e

    def rfunc(x, y):
        result = None
        for p in params:
            amp, xo, yo, major, minor, pa = p
            if result is not None:
                result += elliptical_gaussian(x,y,amp,xo,yo,major,minor,np.radians(pa))
            else:
                result =  elliptical_gaussian(x,y,amp,xo,yo,major,minor,np.radians(pa))
        return result

    return rfunc


def do_mpfit(data, parinfo, B=None):
    """
    Fit multiple gaussian components to data using the information provided by parinfo.
    data may contain 'flagged' or 'masked' data with the value of np.NaN
    input: data - pixel information
           parinfo - initial parameters for mpfit
    return: mpfit object, parameter info
    """

    data = np.array(data)
    mask = np.where(np.isfinite(data))  #the indices of the *non* NaN values in data

    def erfunc(p, fjac=None):
        """The difference between the model and the data"""
        f = ntwodgaussian_mpfit(p)
        model = f(*mask)
        if B is None:
            return [0, model - data[mask]]
        else:
            return [0, (model - data[mask]).dot(B)]

    mp = mpfit(erfunc, parinfo=parinfo, quiet=True)
    mp.dof = len(np.ravel(mask)) - len(parinfo)
    return mp, parinfo


def do_lmfit2(data, params, B=None, D=2, dojac=False):
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

    if D==1:
        mask = mask[0]

    def residual(params,mask,data=None):
        if D==2:
            f = ntwodgaussian_lmfit(params)
            model = f(*mask)
        else:
            model = gaussian(mask,params['amp'].value,params['cen'].value,params['sigma'].value)

        if data is None:
            return model
        if B is None:
            return model-data[mask]
        else:
            return (model - data[mask]).dot(B)
    if dojac:
        if D==1:
            jfn = jacobian
        else:
            jfn = jacobian2d
        result = lmfit.minimize(residual, params, args=(mask,data,),Dfun=jfn)
    else:
        result = lmfit.minimize(residual, params, args=(mask,data,))
    return result, params


def theta_limit(theta):
    """
    Position angle is periodic with period 180\deg
    Constrain pa such that -90<theta<=90
    """
    while theta <= -90:
        theta += 180
    while theta > 90:
        theta -= 180
    return theta


def Cmatrix1d(x,sigma):
    return np.vstack( [ gaussian(x,1., i, 1.*sigma) for i in x ])

Cmatrix2d = Cmatrix

def test_cmatrix():
    nx = 15
    ny = 12
    x,y = np.where(np.ones((nx,ny))==1)

    smoothing = 1.27 # 3pix/beam
    C = Cmatrix(x,y,smoothing,smoothing,0)
    B = Bmatrix(C)
    from matplotlib import pyplot
    fig=pyplot.figure(1)#, figsize=(8,12))
    cmap = pyplot.cm.cubehelix
    cmap.set_bad('y',1.)
    kwargs = {'interpolation':'nearest','cmap':cmap,'origin':'upper'}

    ax = fig.add_subplot(2,2,1)
    ax.imshow(C,**kwargs)
    ax.set_title('C')
    rmlabels(ax)

    ax = fig.add_subplot(2,2,2)
    ax.imshow(inv(C),**kwargs)
    ax.set_title('inv(C)')
    rmlabels(ax)

    ax = fig.add_subplot(2,2,3)
    ax.imshow((inv(C)-B.dot(np.transpose(B)))/inv(C),**kwargs)
    ax.set_title('(inv(C) = B.Bt)/inv(C)')
    rmlabels(ax)

    ax = fig.add_subplot(2,2,4)
    ax.imshow(B.dot(np.transpose(B)),**kwargs)
    ax.set_title('B.Bt')
    rmlabels(ax)

    pyplot.show()

def jacobian(pars,x,data=None):
    amp = pars['amp'].value
    cen = pars['cen'].value
    sigma = pars['sigma'].value

    model = gaussian(x,amp,cen,sigma)

    dmds = model/amp
    dmdcen = model/sigma**2*(x-cen)
    dmdsigma = model*(x-cen)**2/sigma**3
    matrix = np.vstack((dmds,dmdcen,dmdsigma))
    matrix = np.transpose(matrix)
    return matrix


def jacobian2d(pars,xy,data=None,emp=True,errs=None):
    x,y = xy
    if emp:
        return emp_jacobian(pars, x, y, errs)

    amp = pars['c0_amp'].value
    xo = pars['c0_xo'].value
    yo = pars['c0_yo'].value
    sx = pars['c0_sx'].value
    sy = pars['c0_sy'].value
    theta  = pars['c0_theta'].value

    # all derivatives are proportional to the model so calculate it first
    model = elliptical_gaussian(x,y, amp,xo, yo, sx, sy, theta)

    if emp:
        # empirical derivatives
        eps = 1e-5
        dmds = elliptical_gaussian(x,y, amp+eps,xo, yo, sx, sy, theta) - model
        dmdxo = elliptical_gaussian(x,y, amp,xo+eps, yo, sx, sy, theta) - model
        dmdyo = elliptical_gaussian(x,y, amp,xo, yo+eps, sx, sy, theta) - model
        dmdsx = elliptical_gaussian(x,y, amp,xo, yo, sx+eps, sy, theta) - model
        dmdsy = elliptical_gaussian(x,y, amp,xo, yo, sx, sy+eps, theta) - model
        dmdtheta = elliptical_gaussian(x,y, amp,xo, yo, sx, sy, theta+eps) - model
        matrix = np.array([dmds,dmdxo,dmdyo,dmdsx,dmdsy,dmdtheta])/eps
        if errs is not None:
            matrix /=errs#**2
        matrix = np.transpose(matrix)
        return matrix

    # precompute for speed
    sint = np.sin(theta)
    cost = np.cos(theta)
    x,y = xy
    xxo = x-xo
    yyo = x-yo
    xcos, ycos, xsin, ysin = cost*xxo, cost*yyo, sint*xxo, sint*yyo

    dmds = model/amp

    dmdxo = cost * (xcos + ysin) /sx**2 + sint* (xsin - ycos) /sy**2
    dmdxo *= model

    dmdyo = sint * (xcos + ysin) /sx**2 - cost * (xsin - ycos) /sy**2
    dmdyo *= model

    dmdsx = model / sx**3 * (xcos + ysin)**2
    dmdsy = model / sy**3 * (xsin - ycos)**2

    dmdtheta = model * (sx**2 - sy**2) * (xsin + ycos) * (xcos + ysin) / sx**2/sy**2

    matrix = np.vstack((dmds,dmdxo,dmdyo,dmdsx,dmdsy,dmdtheta))
    matrix = np.transpose(matrix)
    return matrix


def print_mat(m):
    print m.shape
    for i in m:
        for j in i:
            print "{0:3.1e}".format(j),
        print


def rmlabels(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def test1d():
    nx = 15
    x = np.arange(nx)

    smoothing = 1.5
    # setup the fitting
    # params = lmfit.Parameters()
    # params.add('amp',value=1, min=0.5, max=2)
    # params.add('cen',value=1.*nx/2+0.2, min= nx/4., max = 3.*nx/4)
    # params.add('sigma', value=1*smoothing, min=0.5*smoothing, max=2*smoothing)
    #
    # signal = gaussian(x,params['amp'].value,params['cen'].value,params['sigma'].value)
    snr = 10

    diffs_nocorr = []
    errs_nocorr = []
    diffs_corr = []
    errs_corr = []
    crb_corr = []

    for j in xrange(3):

        params = lmfit.Parameters()
        params.add('amp',value=1, min=0.5, max=2)
        params.add('cen',value=1.*nx/2)
        params.add('sigma', value=1.*smoothing)

        signal = gaussian(x,params['amp'].value,params['cen'].value,params['sigma'].value)
        np.random.seed(1234567+j)
        noise = np.random.random(nx)
        noise = gaussian_filter1d(noise, sigma=smoothing)
        noise -= np.mean(noise)
        noise /= np.std(noise)*snr



        data = signal+noise
        result,fit_params = do_lmfit2(data, params, D=1)
        C = Cmatrix(x,smoothing)
        B = Bmatrix(C)
        result,corr_fit_params = do_lmfit2(data, params, D=1, B=B)

        if np.all( [fit_params[i].stderr >0 for i in fit_params.valuesdict().keys()]):
            diffs_nocorr.append([ params[i].value -fit_params[i].value for i in fit_params.valuesdict().keys()])
            errs_nocorr.append( [fit_params[i].stderr for i in fit_params.valuesdict().keys()])

        # print_par(corr_fit_params)
        if np.all( [corr_fit_params[i].stderr >0 for i in corr_fit_params.valuesdict().keys()]):
            diffs_corr.append([ params[i].value -corr_fit_params[i].value for i in corr_fit_params.valuesdict().keys()])
            errs_corr.append( [corr_fit_params[i].stderr for i in corr_fit_params.valuesdict().keys()])
            crb_corr.append( CRB_errs(jacobian(corr_fit_params,x), C))


    diffs_nocorr = np.array(diffs_nocorr)
    errs_nocorr = np.array(errs_nocorr)
    diffs_corr = np.array(diffs_corr)
    errs_corr = np.array(errs_corr)
    crb_corr = np.array(crb_corr)

    many = j>10

    if not many:
        model = gaussian(x, fit_params['amp'].value, fit_params['cen'].value, fit_params['sigma'].value)
        corr_model = gaussian(x,corr_fit_params['amp'].value, corr_fit_params['cen'].value, corr_fit_params['sigma'].value)
        print "init ",
        print_par(params)
        print "model",
        print_par(fit_params)
        print "corr_model",
        print_par(corr_fit_params)


        from matplotlib import pyplot
        fig=pyplot.figure(1)
        for k, m in enumerate([model,corr_model]):
            ax = fig.add_subplot(1,2,k+1)
            ax.plot(x,signal,'bo',label='Signal')
            ax.plot(x,noise,'k--',label='Noise')
            ax.plot(x,data,'r-',label="Data")
            ax.plot(x,m,'g-',label="Model")
            ax.plot(x,data-m,'go',label="Data-Model")
            ax.legend()
        pyplot.show()
    else:
        from matplotlib import pyplot
        fig = pyplot.figure(2)
        ax = fig.add_subplot(121)
        ax.plot(diffs_nocorr[:,0], label='amp')
        ax.plot(diffs_nocorr[:,1], label='cen')
        ax.plot(diffs_nocorr[:,2], label='sigma')
        ax.set_xlabel("No Corr")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(diffs_corr[:,0], label='amp')
        ax.plot(diffs_corr[:,1], label='cen')
        ax.plot(diffs_corr[:,2], label='sigma')
        ax.set_xlabel("Corr")
        ax.legend()

        print "-- no corr --"
        for i,val in enumerate(fit_params.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}".format(val,np.mean(diffs_nocorr[:,i]), np.std(diffs_nocorr[:,i]), np.mean(errs_nocorr[:,i]))

        print "--  corr --"
        for i,val in enumerate(corr_fit_params.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}, mean(crb_err) {4}".format(val,np.mean(diffs_corr[:,i]),np.std(diffs_corr[:,i]), np.mean(errs_corr[:,i]),np.mean(crb_corr[:,i]))
        # jac = jacobian(corr_fit_params,x)
        # print_mat(jac)
        # print_mat(C)
        # print_mat(B)
        # print CRB_errs(jac,C)
        # print CRB_errs(jac,C,B)
        pyplot.show()


def test2d():
    nx = 15
    ny = 12
    x,y = np.where(np.ones((nx,ny))==1)

    #smoothing = 1.27 # 3pix/beam
    #smoothing = 2.12 # 5pix/beam
    smoothing = 1.5 # ~4.2pix/beam

    snr = 50

    diffs_nocorr = []
    errs_nocorr = []
    crb_nocorr = []
    diffs_corr = []
    errs_corr = []
    crb_corr = []

    nj = 50
    # The model parameters
    params = lmfit.Parameters()
    params.add('c0_amp', value=1, min=0.5, max=2)
    params.add('c0_xo', value=1.*nx/2, min=nx/2.-smoothing/2., max=nx/2.+smoothing/2)
    params.add('c0_yo', value=1.*ny/2, min=ny/2.-smoothing/2., max=ny/2.+smoothing/2.)
    params.add('c0_sx', value=2*smoothing, min=0.8*smoothing)
    params.add('c0_sy', value=smoothing, min=0.8*smoothing)
    params.add('c0_theta',value=16)#, min=-2*np.pi, max=2*np.pi)
    params.components=1

    signal = elliptical_gaussian(x, y,
                                 params['c0_amp'].value,
                                 params['c0_xo'].value,
                                 params['c0_yo'].value,
                                 params['c0_sx'].value,
                                 params['c0_sy'].value,
                                 params['c0_theta'].value).reshape(nx,ny)

    for j in xrange(nj):
        np.random.seed(1234567+j)

        # The initial guess at the parameters
        init_params = copy.deepcopy(params)
        init_params['c0_amp'].value += 0.05* 2*(np.random.random()-0.5)
        init_params['c0_xo'].value += 1*(np.random.random()-0.5)
        init_params['c0_yo'].value += 1*(np.random.random()-0.5)
        init_params['c0_sx'].value = smoothing*1.01
        init_params['c0_sy'].value = smoothing
        init_params['c0_theta'].value = 0
        noise = np.random.random((nx,ny))
        noise = gaussian_filter(noise, sigma=smoothing)
        noise -= np.mean(noise)
        noise /= np.std(noise)*snr

        data = signal + noise
        snrmask = np.where(data < 4/snr)
        #cmask = np.where(signal < 0.5)
        data[snrmask] = np.nan
        #data[cmask] = np.nan
        mx,my = np.where(np.isfinite(data))
        if len(mx)<7:
            continue

        C = Cmatrix(mx,my,smoothing,smoothing,0)
        B = Bmatrix(C)
        errs = 1./snr

        result, fit_params = do_lmfit(data,init_params)#, errs=errs)
        corr_result,corr_fit_params = do_lmfit(data, init_params, B=B)#, errs=errs)

        if np.all( [fit_params[i].stderr >0 for i in fit_params.valuesdict().keys()]):
            if fit_params['c0_sy'].value>fit_params['c0_sx'].value:
                fit_params['c0_sx'],fit_params['c0_sy'] = fit_params['c0_sy'],fit_params['c0_sx']
                fit_params['c0_theta'].value += 90
            fit_params['c0_theta'].value = theta_limit(fit_params['c0_theta'].value)
            diffs_nocorr.append([ params[i].value -fit_params[i].value for i in fit_params.valuesdict().keys()])
            J = emp_jacobian(fit_params,mx,my,errs = errs)
            #errs_nocorr.append(np.sqrt(np.diag(inv(np.transpose(J).dot(J)))))
            errs_nocorr.append( [fit_params[i].stderr for i in fit_params.valuesdict().keys()])
            #crb_nocorr.append( CRB_errs(jacobian2d(fit_params,(mx,my),emp=True,errs=errs),C) )
            crb_nocorr.append( CRB_errs(J,C) )

        if np.all( [corr_fit_params[i].stderr >0 for i in corr_fit_params.valuesdict().keys()]):
            if corr_fit_params['c0_sy'].value>corr_fit_params['c0_sx'].value:
                corr_fit_params['c0_sx'],corr_fit_params['c0_sy'] = corr_fit_params['c0_sy'],corr_fit_params['c0_sx']
                corr_fit_params['c0_theta'].value += 90
            corr_fit_params['c0_theta'].value = theta_limit(corr_fit_params['c0_theta'].value)
            diffs_corr.append([ params[i].value -corr_fit_params[i].value for i in corr_fit_params.valuesdict().keys()])
            J = emp_jacobian(corr_fit_params,mx,my, errs=errs, B=B)
            errs_corr.append( np.sqrt(np.diag(inv(np.transpose(J).dot(J)))))
            #errs_corr.append( [corr_fit_params[i].stderr for i in corr_fit_params.valuesdict().keys()])
            #crb_corr.append( CRB_errs(jacobian2d(corr_fit_params,(mx,my),emp=True,errs=errs), C) )
            crb_corr.append( CRB_errs(emp_jacobian(corr_fit_params,mx,my, errs=errs), C) )

        if nj<10:
            print "init      ",
            print_par(params)
            print "model     ",
            print_par(fit_params)
            print "corr_model",
            print_par(corr_fit_params)
            print

    diffs_nocorr = np.array(diffs_nocorr)
    errs_nocorr = np.array(errs_nocorr)
    crb_nocorr = np.array(crb_nocorr)
    diffs_corr = np.array(diffs_corr)
    errs_corr = np.array(errs_corr)
    crb_corr = np.array(crb_corr)

    many = j>10

    if True:
        print "-- no corr --"
        for i,val in enumerate(fit_params.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, median(err) {3}, median(crb_err) {4}".format(val,np.median(diffs_nocorr[:,i]), np.std(diffs_nocorr[:,i]), np.median(errs_nocorr[:,i]),np.median(crb_nocorr[:,i]))

        print "--  corr --"
        for i,val in enumerate(corr_fit_params.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, median(err) {3}, median(crb_err) {4}".format(val,np.median(diffs_corr[:,i]),np.std(diffs_corr[:,i]), np.median(errs_corr[:,i]),np.median(crb_corr[:,i]))
        print 1./snr

        model =  elliptical_gaussian(x, y,
                                     fit_params['c0_amp'].value,
                                     fit_params['c0_xo'].value,
                                     fit_params['c0_yo'].value,
                                     fit_params['c0_sx'].value,
                                     fit_params['c0_sy'].value,
                                     fit_params['c0_theta'].value).reshape(nx,ny)
        corr_model = elliptical_gaussian(x, y,
                                     corr_fit_params['c0_amp'].value,
                                     corr_fit_params['c0_xo'].value,
                                     corr_fit_params['c0_yo'].value,
                                     corr_fit_params['c0_sx'].value,
                                     corr_fit_params['c0_sy'].value,
                                     corr_fit_params['c0_theta'].value).reshape(nx,ny)
        print "init      ",
        print_par(params)
        print "model     ",
        print_par(fit_params)
        print "corr_model",
        print_par(corr_fit_params)


        from matplotlib import pyplot
        fig=pyplot.figure(1)#, figsize=(8,12))
        # This sets all nan pixels to be a nasty yellow colour
        cmap = pyplot.cm.cubehelix
        cmap.set_bad('y',1.)
        kwargs = {'interpolation':'nearest','cmap':cmap,'vmin':-0.1,'vmax':1, 'origin':'lower'}

        ax = fig.add_subplot(3,3,1)
        ax.imshow(signal,**kwargs)
        ax.set_title('True')
        rmlabels(ax)

        ax = fig.add_subplot(3,3,2)
        ax.imshow(signal+noise,**kwargs)
        ax.set_title('Data')
        rmlabels(ax)

        ax = fig.add_subplot(3,3,3)
        ax.imshow(noise,**kwargs)
        ax.set_title("Noise")
        rmlabels(ax)

        ax = fig.add_subplot(3,2,3)
        ax.imshow(model,**kwargs)
        ax.set_title('Model')
        rmlabels(ax)

        ax = fig.add_subplot(3,2,4)
        ax.imshow(corr_model,**kwargs)
        ax.set_title('Corr_Model')
        rmlabels(ax)

        ax = fig.add_subplot(3,2,5)
        ax.imshow(data - model, **kwargs)
        ax.set_title('Data - Model')
        rmlabels(ax)

        # covar = corr_result.covar
        # J = emp_jacobian(fit_params,mx,my)
        # JJ = inv(np.transpose(J).dot(inv(C)).dot(J))
        # #print covar
        # #print np.diag(covar)
        # print np.sqrt(np.diag(covar))
        # #print JJ
        # #print np.diag(JJ)
        # print np.sqrt(np.diag(JJ))
        # print CRB_errs(J,C)
        # print corr_fit_params
        # sys.exit()
        ax = fig.add_subplot(3,2,6)
        mappable = ax.imshow(data - corr_model, **kwargs)
        ax.set_title('Data - Corr_model')
        rmlabels(ax)

        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(mappable, cax=cax)

        fig = pyplot.figure(2)
        ax = fig.add_subplot(121)
        ax.plot(diffs_nocorr[:,0], label='amp')
        ax.plot(diffs_nocorr[:,1], label='xo')
        ax.plot(diffs_nocorr[:,2], label='yo')
        ax.plot(diffs_nocorr[:,3], label='sx')
        ax.plot(diffs_nocorr[:,4], label='sy')
        ax.plot(diffs_nocorr[:,5]/180, label='theta/180')
        ax.set_xlabel("No Corr")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(diffs_corr[:,0], label='amp')
        ax.plot(diffs_corr[:,1], label='xo')
        ax.plot(diffs_corr[:,2], label='yo')
        ax.plot(diffs_corr[:,3], label='sx')
        ax.plot(diffs_corr[:,4], label='sy')
        ax.plot(diffs_corr[:,5]/180, label='theta/180')
        ax.set_xlabel("Corr")
        ax.legend()

        hkwargs = {'histtype':'step','bins':51,'range':(-1,1)}
        fig = pyplot.figure(3)
        ax = fig.add_subplot(121)
        ax.hist(diffs_nocorr[:,0], label='amp',**hkwargs)
        ax.hist(diffs_nocorr[:,1], label='xo',**hkwargs)
        ax.hist(diffs_nocorr[:,2], label='yo',**hkwargs)
        ax.hist(diffs_nocorr[:,3], label='sx',**hkwargs)
        ax.hist(diffs_nocorr[:,4], label='sy',**hkwargs)
        ax.hist(diffs_nocorr[:,5]/180, label='theta/180',**hkwargs)
        ax.set_xlabel("No Corr")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.hist(diffs_corr[:,0], label='amp',**hkwargs)
        ax.hist(diffs_corr[:,1], label='xo',**hkwargs)
        ax.hist(diffs_corr[:,2], label='yo',**hkwargs)
        ax.hist(diffs_corr[:,3], label='sx',**hkwargs)
        ax.hist(diffs_corr[:,4], label='sy',**hkwargs)
        ax.hist(diffs_corr[:,5]/180, label='theta/180',**hkwargs)
        ax.set_xlabel("Corr")
        ax.legend()

        # jac = jacobian2d(corr_fit_params,(x,y),emp=True,errs=errs)
        # print_mat(jac[:10,:10])
        # print_mat(C[:10,:10])
        # print_mat(B[:10,:10])
        # print CRB_errs(jac,C)
        # print CRB_errs(jac,C,B)
        pyplot.show()


def test_two_components():

    nx = 20
    ny = 20
    x,y = np.where(np.ones((nx,ny))==1)

    #smoothing = 1.27 # 3pix/beam
    smoothing = 2.12 # 5pix/beam
    #smoothing = 1.5 # ~4.2pix/beam

    snr = 500

    # The model parameters
    params = lmfit.Parameters()
    params.add('c0_amp', value=1, min=0.5, max=2)
    params.add('c0_xo', value=1.*nx/3, min=nx/3.-smoothing/2., max=nx/3.+smoothing/2.)
    params.add('c0_yo', value=1.*ny/3, min=ny/3.-smoothing/2., max=ny/3.+smoothing/2.)
    params.add('c0_sx', value=2*smoothing, min=0.8*smoothing)
    params.add('c0_sy', value=smoothing, min=0.8*smoothing)
    params.add('c0_theta',value=45)
    params.components = 1
    if True:
        vary=True
        params.add('c1_amp', value=0.9, min=0, max=3, vary=vary)
        params.add('c1_xo', value=2.*nx/3, min=2*nx/3.-smoothing/2., max=2*nx/3.+smoothing/2., vary=vary)
        params.add('c1_yo', value=2.*ny/3, min=2*ny/3.-smoothing/2., max=2*ny/3.+smoothing/2., vary=vary)
        params.add('c1_sx', value=2*smoothing, min=0.8*smoothing, vary=vary)
        params.add('c1_sy', value=smoothing, min=0.8*smoothing, vary=vary)
        params.add('c1_theta',value=0, vary=vary)
        params.components=2

    signal = ntwodgaussian_lmfit(params)(x,y).reshape(nx,ny)

    np.random.seed(1234567)

    # The initial guess at the parameters
    init_params = copy.deepcopy(params)
    for i in range(params.components):
        prefix = 'c{0}_'.format(i)
        init_params[prefix+'amp'].value += 0.05
        init_params[prefix+'xo'].value += 0.01
        init_params[prefix+'yo'].value += 0.01
        init_params[prefix+'sx'].value = smoothing *(1+(2*i+1)/100.)
        init_params[prefix+'sy'].value = smoothing *(1+(2*i+2)/100.)
        init_params[prefix+'theta'].value = i

    noise = np.random.random((nx,ny))
    noise = gaussian_filter(noise, sigma=smoothing)
    noise -= np.mean(noise)
    noise /= np.std(noise)*snr

    data = signal + noise
    #snrmask = np.where(data < 4/snr)
    #cmask = np.where(signal < 0.5)
    #data[snrmask] = np.nan
    #data[cmask] = np.nan
    mx,my = np.where(np.isfinite(data))

    C = Cmatrix(mx,my,smoothing,smoothing,0)
    B = Bmatrix(C)
    #errs = 1./snr

    result, fit_params = do_lmfit(data, init_params, dojac=True)#, errs=errs)

    corr_result,corr_fit_params = do_lmfit(data, init_params, B=B, dojac=True)#, errs=errs)

    print "true    "
    print_par(params)
    print "initial "
    print_par(init_params)
    print "nocorr  ", np.mean(result.residual), np.std(result.residual)
    print_par(fit_params)

    print "+corr   ", np.mean(corr_result.residual), np.std(corr_result.residual)
    print_par(corr_fit_params)

    if True:
        from matplotlib import pyplot
        fig=pyplot.figure(1)
        cmap = pyplot.cm.cubehelix
        cmap.set_bad('y',1.)
        kwargs = {'interpolation':'nearest','cmap':cmap,'vmin':-0.1,'vmax':2, 'origin':'lower'}
        ax = fig.add_subplot(2,2,1)
        ax.imshow(signal, **kwargs)
        ax.set_title("signal")
        rmlabels(ax)

        ax = fig.add_subplot(2,2,2)
        ax.imshow(ntwodgaussian_lmfit(init_params)(x,y).reshape(nx,ny),**kwargs)
        ax.set_title("initial")
        rmlabels(ax)

        ax = fig.add_subplot(2,2,3)
        ax.imshow(ntwodgaussian_lmfit(fit_params)(x,y).reshape(nx,ny),**kwargs)
        ax.set_title("-corr")
        rmlabels(ax)

        ax = fig.add_subplot(2,2,4)
        ax.imshow(ntwodgaussian_lmfit(corr_fit_params)(x,y).reshape(nx,ny),**kwargs)
        ax.set_title("+corr")
        rmlabels(ax)

        pyplot.show()



def test_jacobian():
    nx = 15
    ny = 12
    x,y = np.where(np.ones((nx,ny))==1)

    smoothing = 1.27 # 3pix/beam
    #smoothing = 2.12 # 5pix/beam
    #smoothing = 1.5 # ~4.2pix/beam

    snr = 10
    nj = 10
    # The model parameters
    params = lmfit.Parameters()
    params.add('c0_amp', value=1, min=0.5, max=2)
    params.add('c0_xo', value=1.*nx/2, min=nx/2.-smoothing/2., max=nx/2.+smoothing/2)
    params.add('c0_yo', value=1.*ny/2, min=ny/2.-smoothing/2., max=ny/2.+smoothing/2.)
    params.add('c0_sx', value=2*smoothing, min=0.8*smoothing)
    params.add('c0_sy', value=smoothing, min=0.8*smoothing)
    params.add('c0_theta',value= 45)
    params.components=1
    if True:
        vary=True
        params.add('c1_amp', value=0.9, min=0, max=3, vary=vary)
        params.add('c1_xo', value=2.*nx/3, min=2*nx/3.-smoothing/2., max=2*nx/3.+smoothing/2., vary=vary)
        params.add('c1_yo', value=2.*ny/3, min=2*ny/3.-smoothing/2., max=2*ny/3.+smoothing/2., vary=vary)
        params.add('c1_sx', value=2*smoothing, min=0.8*smoothing, vary=vary)
        params.add('c1_sy', value=smoothing, min=0.8*smoothing, vary=vary)
        params.add('c1_theta',value=0, vary=vary)
        params.components=2


    from matplotlib import pyplot
    fig=pyplot.figure(1)#, figsize=(8,12))
    # This sets all nan pixels to be a nasty yellow colour
    cmap = pyplot.cm.cubehelix
    cmap.set_bad('y',1.)
    kwargs = {'interpolation':'nearest','cmap':cmap,'vmin':-0.1,'vmax':1, 'origin':'lower'}

    C = Cmatrix(x, y, smoothing, smoothing,0)
    B = Bmatrix(C)
    jdata = emp_jacobian(params, x, y)
    jdata = ana_jacobian(params, x, y)
    for i,p in enumerate(['amp','xo','yo','sx','sy','theta']*params.components):

        ax = fig.add_subplot(3,6,i+1)
        ax.imshow(jdata[:,i].reshape(nx,ny),**kwargs)
        ax.set_title(p)
        rmlabels(ax)
    ax = fig.add_subplot(3,6,13)
    ax.imshow(ntwodgaussian_lmfit(params)(x,y).reshape(nx,ny), **kwargs)
    ax.set_title("model")
    rmlabels(ax)
    pyplot.show()


def JC_err_comp():
    nx = 15
    ny = 12
    x,y = np.where(np.ones((nx,ny))==1)

    smoothing = 1.27 # 3pix/beam
    #smoothing = 2.12 # 5pix/beam
    #smoothing = 1.5 # ~4.2pix/beam



    condon_err_list_corr= []
    condon_err_list_nocorr= []
    CRB_err_list_corr = []
    CRB_err_list_nocorr = []

    nj = 50
    nsnr = 10
    snr = np.logspace(np.log10(5),np.log10(50),nsnr)
    print snr
    # The model parameters
    params = lmfit.Parameters()
    params.add('c0_amp', value=1, min=0.5, max=2)
    params.add('c0_xo', value=1.*nx/2)
    params.add('c0_yo', value=1.*ny/2)
    params.add('c0_sx', value=2*smoothing, min=0.8*smoothing)
    params.add('c0_sy', value=smoothing, min=0.8*smoothing)
    params.add('c0_theta',value=0, min=-2*np.pi, max=2*np.pi)
    params.components=1

    for k in xrange(nsnr):
        diffs_nocorr = []
        errs_nocorr = []
        crb_nocorr = []
        diffs_corr = []
        errs_corr = []
        crb_corr = []
        print k
        for j in xrange(nj):
            np.random.seed(1234567+j)

            # The initial guess at the parameters
            init_params = copy.deepcopy(params)
            init_params['c0_amp'].value += 0.05* 2*(np.random.random()-0.5)
            init_params['c0_xo'].value += 1*(np.random.random()-0.5)
            init_params['c0_yo'].value += 1*(np.random.random()-0.5)
            init_params['c0_sx'].value = smoothing*1.01
            init_params['c0_sy'].value = smoothing
            init_params['c0_theta'].value = 0

            signal = elliptical_gaussian(x, y,
                                         params['c0_amp'].value,
                                         params['c0_xo'].value,
                                         params['c0_yo'].value,
                                         params['c0_sx'].value,
                                         params['c0_sy'].value,
                                         params['c0_theta'].value).reshape(nx,ny)

            noise = np.random.random((nx,ny))
            noise = gaussian_filter(noise, sigma=smoothing)
            noise -= np.mean(noise)
            noise /= np.std(noise)*snr[k]

            data = signal + noise
            mask = np.where(data < 4/snr[k])
            data[mask] = np.nan
            mx,my = np.where(np.isfinite(data))
            if len(mx)<=7:
                continue

            result, fit_params = do_lmfit(data,init_params)

            C = Cmatrix2d(mx,my,smoothing,smoothing,0)
            B = Bmatrix(C)
            corr_result,corr_fit_params = do_lmfit(data, init_params, B=B)
            errs = np.ones(C.shape[0],dtype=np.float32)/snr[k]
            I = np.identity(C.shape[0])

            if np.all( [fit_params[i].stderr >0 for i in fit_params.valuesdict().keys()]):
                if fit_params['c0_sy'].value>fit_params['c0_sx'].value:
                    fit_params['c0_sx'],fit_params['c0_sy'] = fit_params['c0_sy'],fit_params['c0_sx']
                    fit_params['c0_theta'].value += np.pi/2
                fit_params['c0_theta'].value = theta_limit(fit_params['c0_theta'].value)
                diffs_nocorr.append([ params[i].value -fit_params[i].value for i in fit_params.valuesdict().keys()])
                errs_nocorr.append( [fit_params[i].stderr for i in fit_params.valuesdict().keys()])
                crb_nocorr.append( CRB_errs(jacobian2d(fit_params,(mx,my),emp=True,errs=errs),I) )

            if np.all( [corr_fit_params[i].stderr >0 for i in corr_fit_params.valuesdict().keys()]):
                if corr_fit_params['c0_sy'].value>corr_fit_params['c0_sx'].value:
                    corr_fit_params['c0_sx'],corr_fit_params['c0_sy'] = corr_fit_params['c0_sy'],corr_fit_params['c0_sx']
                    corr_fit_params['c0_theta'].value += np.pi/2
                corr_fit_params['c0_theta'].value = theta_limit(corr_fit_params['c0_theta'].value)
                diffs_corr.append([ params[i].value -corr_fit_params[i].value for i in corr_fit_params.valuesdict().keys()])
                errs_corr.append( [corr_fit_params[i].stderr for i in corr_fit_params.valuesdict().keys()])
                crb_corr.append( CRB_errs(jacobian2d(corr_fit_params,(mx,my),emp=True,errs=errs), C) )

            if nj<10:
                print "init ",
                print_par(params)
                print "model",np.std(result.residual),
                print_par(fit_params)
                print "corr_model", np.std(corr_result.residual),
                print_par(corr_fit_params)

        diffs_nocorr = np.array(diffs_nocorr)
        errs_nocorr = np.array(errs_nocorr)
        crb_nocorr = np.array(crb_nocorr)
        diffs_corr = np.array(diffs_corr)
        errs_corr = np.array(errs_corr)
        crb_corr = np.array(crb_corr)

        CRB_err_list_corr.append(np.median(crb_corr[:,0]*snr[k]))
        CRB_err_list_nocorr.append(np.median(crb_nocorr[:,0]*snr[k]))

    from matplotlib import pyplot
    fig = pyplot.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(snr, CRB_err_list_corr, 'ro-', label='$C^{-1}$')
    ax.plot(snr, CRB_err_list_nocorr, 'bo-', label='$C = I$')
    ax.set_xlabel("SNR")
    ax.set_ylabel("Peak Flux Error ($\sigma$)")
    ax.set_xlim(snr[0],snr[-1])
    ax.set_ylim((0,1.2))
    ax.legend()

    pyplot.show()


def test2d_load():
    from astropy.io import fits
    #from aegean import do_lmfit
    data = fits.open('../lm_dev/test.fits')[0].data
    params = lmfit.Parameters()
    params.add('amp', value=10.)
    params.add('xo', value=4.01)
    params.add('yo', value=7.)
    params.add('sx', value=4.01,min=1)
    params.add('sy', value=2,min=1)
    params.add('theta',value=np.pi/4)
    params.components=1

    result, fit_params = do_lmfit2(data,params,dojac=True)
    #print result.residual
    print np.mean(result.residual), np.std(result.residual)
    print_par(fit_params)

    from matplotlib import pyplot
    fig=pyplot.figure(1, figsize=(8,12))
    kwargs = {'interpolation':'nearest','cmap':pyplot.cm.cubehelix,'vmin':-2,'vmax':10, 'origin':'lower'}

    ax = fig.add_subplot(5,3,1)
    ax.imshow(data,**kwargs)
    ax.set_title('Data')


    ax = fig.add_subplot(5,3,2)
    ax.imshow(data + result.residual.reshape(data.shape),**kwargs)
    ax.set_title('model')

    ax = fig.add_subplot(5,3,3)
    ax.imshow(result.residual.reshape(data.shape),**kwargs)
    ax.set_title('residual')


    xy = np.where(np.isfinite(data))
    jac = jacobian2d(fit_params,xy,emp=False)
    shape = data.shape

    print jac.shape
    for i,name in enumerate(['dmds','dmdxo','dmdyo','dmdsx','dmdsy','dmdtheta']):
        print i,name
        ax = fig.add_subplot(5,3,i+4)
        ax.imshow(jac[:,i].reshape(shape),**kwargs)
        ax.set_title(name)

    jac = jacobian2d(fit_params,xy,emp=True)

    for i,name in enumerate(['dmds','dmdxo','dmdyo','dmdsx','dmdsy','dmdtheta']):
        ax = fig.add_subplot(5,3,i+10)
        ax.imshow(jac[:,i].reshape(shape),**kwargs)
        ax.set_title(name)


    pyplot.show()


def test_CRB():

    def model(x,a,b):
        return a*x+b

    def jac(x,a,b):
        m = model(x,a,b)
        eps = 1e-6
        dmda = model(x,a+eps,b) - m
        dmdb = model(x,a,b+eps) - m
        return np.transpose(np.array( [dmda,dmdb])/eps / errs**2)

    x = np.array([0,10])
    errs = np.array([0.1,0.1])
    C = np.identity(len(x))
    p0=[1,0]
    print model(x,*p0)
    print jac(x,*p0)
    print CRB_errs(jac(x,*p0),C)



# @profile
def test_lm(data):

    x,y = np.meshgrid(range(data.shape[0]),range(data.shape[1]))
    # convert 2d data into 1d lists, masking out nans in the process
    data, mask, shape = ravel_nans(data)
    x = np.ravel(x[mask])
    y = np.ravel(y[mask])

    # setup the fitting
    g1 = lmfit.Model(two_d_gaussian,independent_vars=['x','y'],prefix="c1_") 
    g1.set_param_hint('amp',value=1)
    g1.set_param_hint('xo',value=1.2) 
    g1.set_param_hint('yo',value=1)
    g1.set_param_hint('major', value=2, min=1, max=4)
    g1.set_param_hint('minor', value=1, min=0.5, max=3)
    g1.set_param_hint('pa', value = 0 , min=-math.pi, max=math.pi)

    g2 = lmfit.Model(two_d_gaussian,independent_vars=['x','y'],prefix="c2_") 
    g2.set_param_hint('amp',value=1)
    g2.set_param_hint('xo',value=3) 
    g2.set_param_hint('yo',value=3.1)
    g2.set_param_hint('major', value=2, min=0.5, max=2)
    g2.set_param_hint('minor', value=1, min=0.5, max=2)
    g2.set_param_hint('pa', value = 0 , min=-math.pi, max=math.pi)

    #do the fit
    gmod = reduce(lambda x,y: x+y,[g1+g2])
    params = gmod.make_params()
    result = gmod.fit(data,x=x,y=y,params=params)
    return unravel_nans(result.best_fit,mask,shape),result.values

# @profile
def test_mpfit(data):
    i=1
    parinfo=[]
    parinfo.append( {'value':1,
       'fixed':False,
       'parname':'{0}:amp'.format(i),
       'limits':[0,2],
       'limited':[False,False]} )
    parinfo.append( {'value':1.2,
       'fixed':False,
       'parname':'{0}:xo'.format(i),
       'limits':[0,0],
       'limited':[False,False]} )
    parinfo.append( {'value':1,
       'fixed':False,
       'parname':'{0}:yo'.format(i),
       'limits':[0,0],
       'limited':[False,False]} )
    parinfo.append( {'value':2,
       'fixed': False,
       'parname':'{0}:major'.format(i),
       'limits':[1,4],
       'limited':[True,True]} )
    parinfo.append( {'value':1,
       'fixed': False,
       'parname':'{0}:minor'.format(i),
       'limits':[0.5,3],
       'limited':[True,True]} )
    parinfo.append( {'value':0,
       'fixed': False,
       'parname':'{0}:pa'.format(i),
       'limits':[-180,180],
       'limited':[False,False]} )
    i=2
    parinfo.append( {'value':1,
       'fixed':False,
       'parname':'{0}:amp'.format(i),
       'limits':[0,2],
       'limited':[False,False]} )
    parinfo.append( {'value':3,
       'fixed':False,
       'parname':'{0}:xo'.format(i),
       'limits':[0,0],
       'limited':[False,False]} )
    parinfo.append( {'value':3.1,
       'fixed':False,
       'parname':'{0}:yo'.format(i),
       'limits':[0,0],
       'limited':[False,False]} )
    parinfo.append( {'value':2,
       'fixed': False,
       'parname':'{0}:major'.format(i),
       'limits':[0.5,2],
       'limited':[True,True]} )
    parinfo.append( {'value':1,
       'fixed': False,
       'parname':'{0}:minor'.format(i),
       'limits':[0.5,2],
       'limited':[True,True]} )
    parinfo.append( {'value':0,
       'fixed': False,
       'parname':'{0}:pa'.format(i),
       'limits':[-180,180],
       'limited':[False,False]} )
    mp, parinfo = multi_gauss(data,parinfo)

    #return the data
    #print parinfo
    inpars = [a['value'] for a in parinfo]
    #print inpars
    ret = ntwodgaussian(inpars)(*np.indices(data.shape))
    print ret.shape
    return ret,inpars


def compare():
    dxy = 20
    dim = np.linspace(0,5,dxy)
    x,y = np.meshgrid(dim,dim)
    ztrue = two_d_gaussian(x,y, 1, 1, 1, 3, 1, 0) + two_d_gaussian(x,y, 1,3,3,1,1,0)
    z = ztrue + 0.4*(0.5-np.random.random((dxy,dxy)))
    z[:10,13:14]=np.nan

    lm_data, lm_pars = test_lm(z)
    mp_data, mp_pars = test_mpfit(z)

    print lm_pars, mp_pars

    from matplotlib import pyplot
    fig=pyplot.figure()
    kwargs = {'interpolation':'nearest','cmap':pyplot.cm.cubehelix,'vmin':-0.1,'vmax':1}
    ax = fig.add_subplot(2,2,1)
    ax.imshow(ztrue,**kwargs)
    ax.set_title('True')

    ax = fig.add_subplot(2,2,2)
    ax.imshow(z,**kwargs)
    ax.set_title('Data')

    ax = fig.add_subplot(2,2,3)
    ax.imshow(lm_data,**kwargs)
    ax.set_title('LM Fit')

    ax = fig.add_subplot(2,2,4)
    ax.imshow(mp_data,**kwargs)
    ax.set_title('MP Fit')

    pyplot.show()
    return

def gmean(indata):
    """
    Calculate the geometric mean of a data set taking account of
    values that may be negative, zero, or nan
    :param data: a list of floats or ints
    :return: the geometric mean of the data
    """
    data = np.ravel(indata)
    if np.inf in data:
        return np.inf, np.inf

    finite = data[np.isfinite(data)]
    if len(finite) < 1:
        return np.nan, np.nan
    #determine the zero point and scale all values to be 1 or greater
    scale = abs(np.min(finite)) + 1
    finite += scale
    #calculate the geometric mean of the scaled data and scale back
    lfinite = np.log(finite)
    flux = np.exp(np.mean(lfinite)) - scale
    error = np.nanstd(lfinite) * flux
    return flux, abs(error)

def test_lm_corr_noise():
    """
    :return:
    """
    nx = 50
    smoothing = 3
    x  = np.arange(nx)
    Ci = cinverse(x,smoothing)

    def residual(pars,x,data=None):
        amp = pars['amp'].value
        cen = pars['cen'].value
        sigma = pars['sigma'].value
        model = gaussian(x, amp, cen, sigma)
        if data is None:
            return model
        resid = (model-data) * Ci # * np.matrix(model-data).T
        return resid.tolist()[0]

    def residual_nocorr(pars,x,data=None):
        amp = pars['amp'].value
        cen = pars['cen'].value
        sigma = pars['sigma'].value
        model = gaussian(x, amp, cen, sigma)
        if data is None:
            return model
        resid = model-data
        return resid

    x = np.arange(nx)

    params = lmfit.Parameters()
    params.add('amp', value=5.0)#, min=9, max=11)
    params.add('cen', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('sigma', value=1.5*smoothing, min=smoothing, max=3.0*smoothing)

    iparams = copy.deepcopy(params)

    signal = gaussian(x, iparams['amp'].value, iparams['cen'].value, iparams['sigma'].value)

    np.random.seed(1234567)
    noise = np.random.random(nx)
    noise = gaussian_filter1d(noise, sigma=smoothing)
    noise -= np.mean(noise)
    noise /= np.std(noise)

    data = signal + noise

    #data, mask, shape = ravel_nans(data)
    mi = lmfit.minimize(residual, params, args=(x,data))
    model = gaussian(x, params['amp'].value, params['cen'].value, params['sigma'].value)
    #data = unravel_nans(data,mask,shape)
    mi2 = lmfit.minimize(residual_nocorr,iparams,args=(x,data))
    model_nocor = gaussian(x, iparams['amp'].value, iparams['cen'].value, iparams['sigma'].value)

    print CRB_errs(dmdtheta(params,x),Ci)
 
    print params
    print iparams
    from matplotlib import pyplot
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,signal, label='signal')
    ax.plot(x,noise, label='noise')
    ax.plot(x,data, label='data')
    ax.plot(x,model, label='model')
    ax.plot(x,model_nocor, label='model_nocor')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel("'flux'")
    pyplot.show()


def print_par(params):
    print ','.join("{0}: {1:5.2f}".format(k,params[k].value) for k in params.valuesdict().keys())


def test_lm1d_errs():
    """
    :return:
    """

    nx = 50
    smoothing = 2.12 # FWHM ~ 5pixels

    x  = 1.*np.arange(nx)
    C = Cmatrix(x,smoothing)
    B = Bmatrix(C)

    def residual(pars,x,data=None):
        amp = pars['amp'].value
        cen = pars['cen'].value
        sigma = pars['sigma'].value
        model = gaussian(x, amp, cen, sigma)
        if data is None:
            return model
        resid = (model-data).dot(B)
        return resid

    def residual_nocorr(pars,x,data=None):
        amp = pars['amp'].value
        cen = pars['cen'].value
        sigma = pars['sigma'].value
        model = gaussian(x, amp, cen, sigma)
        if data is None:
            return model
        resid = model-data
        return resid

    # Create the signal
    s_params = lmfit.Parameters()
    s_params.add('amp', value=50.0)#, min=9, max=11)
    s_params.add('cen', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    s_params.add('sigma', value=smoothing, min=0.5*smoothing, max=2.0*smoothing)
    print_par(s_params)
    signal = gaussian(x,s_params['amp'].value, s_params['cen'].value,s_params['sigma'].value)

    # create the initial guess
    iparams = copy.deepcopy(s_params)
    iparams['amp'].value+=1
    iparams['cen'].value-=1
    iparams['sigma'].value+=0.5


    diffs_corr = []
    errs_corr = []
    diffs_nocorr = []
    errs_nocorr = []
    crb_corr = []

    for n in xrange(100):
        # need to re-init this.
        # print n


        np.random.seed(23423 + n)
        noise = np.random.random(nx)
        noise = gaussian_filter(noise, sigma=smoothing)
        noise -= np.mean(noise)
        noise /= np.std(noise)

        data = signal + noise
        #mask the data
        data[data<3] = np.nan
        data, mask, shape = ravel_nans(data)
        x_mask = x[mask]
        if len(x_mask) <=4:
            continue
        C = Cmatrix(x_mask,smoothing)
        B = Bmatrix(C)

        #print np.max(B.dot(np.transpose(B)) - inv(C))

        pars_corr = copy.deepcopy(iparams)
        mi_corr = lmfit.minimize(residual, pars_corr, args=(x_mask, data))#,Dfun=jacobian)
        pars_nocorr = copy.deepcopy(iparams)
        mi_nocorr = lmfit.minimize(residual_nocorr, pars_nocorr, args=(x_mask, data))#,Dfun=jacobian)


        if np.all( [pars_corr[i].stderr >0 for i in pars_corr.valuesdict().keys()]):
            diffs_corr.append([ s_params[i].value -pars_corr[i].value for i in pars_corr.valuesdict().keys()])
            errs_corr.append( [pars_corr[i].stderr for i in pars_corr.valuesdict().keys()])
            crb_corr.append( CRB_errs(jacobian(s_params,x_mask),C))
        #print mi_corr.nfev, mi_nocorr.nfev
        #print_par(pars_corr)

        if np.all( [pars_nocorr[i].stderr >0 for i in pars_nocorr.valuesdict().keys()]):
            diffs_nocorr.append([ s_params[i].value -pars_nocorr[i].value for i in pars_nocorr.valuesdict().keys()])
            errs_nocorr.append( [pars_nocorr[i].stderr for i in pars_nocorr.valuesdict().keys()])

    diffs_corr = np.array(diffs_corr)
    errs_corr = np.array(errs_corr)
    diffs_nocorr = np.array(diffs_nocorr)
    errs_nocorr = np.array(errs_nocorr)
    crb_corr = np.array(crb_corr)


    many = True
    from matplotlib import pyplot
    if many:
        fig = pyplot.figure(1)
        ax = fig.add_subplot(121)
        ax.plot(diffs_corr[:,0], label='amp')
        ax.plot(diffs_corr[:,1], label='cen')
        ax.plot(diffs_corr[:,2], label='sigma')
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(diffs_nocorr[:,0], label='amp')
        ax.plot(diffs_nocorr[:,1], label='cen')
        ax.plot(diffs_nocorr[:,2], label='sigma')
        ax.legend()

    if not many:
        model_nocor = gaussian(x, pars_nocorr['amp'].value, pars_nocorr['cen'].value, pars_nocorr['sigma'].value)
        model = gaussian(x, pars_corr['amp'].value, pars_corr['cen'].value, pars_corr['sigma'].value)

        fig = pyplot.figure(2)
        ax = fig.add_subplot(111)
        ax.plot(x,signal, label='signal')
        ax.plot(x,noise, label='noise')
        ax.plot(x_mask,data, label='data')
        ax.plot(x,model, label='model')
        ax.plot(x,model_nocor, label='model_nocor')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel("'flux'")

    if False:
        fig = pyplot.figure(3)
        C = np.vstack( [ gaussian(x,1., i, 1.*smoothing) for i in x ])
        #C[C<1e-3]=0.0
        ax = fig.add_subplot(121)
        cb = ax.imshow(C,interpolation='nearest',cmap=pyplot.cm.cubehelix)
        pyplot.colorbar(cb)

        L,Q = eigh(C)
        print L
        S = np.diag(1/np.sqrt(abs(L)))
        B = Q.dot(S)

        ax = fig.add_subplot(122)
        cb = ax.imshow(B,interpolation='nearest',cmap=pyplot.cm.cubehelix)
        pyplot.colorbar(cb)

    if many:
        print " -- corr --"
        for i,val in enumerate(pars_corr.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}, mean(crb_err) {4}".format(val,np.mean(diffs_corr[:,i]),np.std(diffs_corr[:,i]), np.mean(errs_corr[:,i]),np.mean(crb_corr[:,i]))
        print "-- no corr --"
        for i,val in enumerate(pars_nocorr.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}".format(val,np.mean(diffs_nocorr[:,i]), np.std(diffs_nocorr[:,i]), np.mean(errs_nocorr[:,i]))

    pyplot.show()


def test_lm_corr_noise_2d():
    """
    :return:
    """
    from cmath import phase
    nx = 15
    smoothing = 3
    x, y = np.meshgrid(range(nx),range(nx))
    C = np.vstack( [ np.ravel(two_d_gaussian(x,y,1, i, j, smoothing, smoothing, 0))
                             for i,j in zip(x.ravel(),y.ravel())])

    # The square root should give a matrix of real values, so the inverse should all be real
    # Some kind of round off effect stops this from being true so we enforce it.
    Ci = abs(np.matrix(sqrtm(C)).I)
    # Ci = np.matrix(np.diag(np.ones(nx**2)))
    def residual(pars,x,y,data=None):
        amp = pars['amp'].value
        xo = pars['xo'].value
        yo = pars['yo'].value
        major = pars['major'].value
        minor = pars['minor'].value
        pa = pars['pa'].value
        model = np.ravel(two_d_gaussian(x, y, amp, xo, yo, major, minor, pa))
        if data is None:
            return model
        resid = (model-data) * Ci # * np.matrix(model-data).T
        return resid.tolist()[0]

    params = lmfit.Parameters()
    params.add('amp', value=5, min=3, max=7)
    params.add('xo', value=nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('yo', value=nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('major', value=smoothing, min=0.8*smoothing, max=1.2*smoothing)
    params.add('minor', value=smoothing, min=0.8*smoothing, max=1.2*smoothing)
    params.add('pa', value=0)#, min=-1.*np.pi, max=np.pi)


    signal = residual(params, x, y) # returns model
    signal = signal.reshape(nx,nx)

    np.random.seed(1234567)
    noise = np.random.random((nx,nx))
    noise = gaussian_filter(noise, sigma=smoothing)
    noise -= np.mean(noise)
    noise /= np.std(noise)

    data = np.ravel(signal + noise)

    mi = lmfit.minimize(residual, params, args=(x, y, data))
    data = data.reshape(nx,nx)
    model = residual(params, x, y).reshape(nx,nx)
    print params

    kwargs = {'vmin':-1, 'vmax':6, 'interpolation':'nearest'}
    from matplotlib import pyplot
    fig = pyplot.figure()
    ax = fig.add_subplot(221)
    ax.imshow(noise, **kwargs)
    ax.set_title('noise')
    print "noise rms: {0}".format(np.std(noise))

    ax = fig.add_subplot(222)
    ax.imshow(data, **kwargs)
    ax.set_title('data')

    ax = fig.add_subplot(223)
    ax.imshow(model, **kwargs)
    ax.set_title('model')

    ax = fig.add_subplot(224)
    ax.imshow(data-model, **kwargs)
    ax.set_title('residual')
    print "resid rms: {0}".format(np.std(model-data))

    pyplot.show()


def test_lm2d_errs():
    """
    :return:
    """

    nx = 10
    ny = 13
    smoothing = 2#2./(2*np.sqrt(2*np.log(2))) #5 pixels per beam

    # x, y = np.meshgrid(range(nx),range(ny))
    # rx, ry = zip(*[ (x[i,j],y[i,j]) for i in range(len(x)) for j in range(len(y))])
    # rx = np.array(rx)
    # ry = np.array(ry)
    x, y = np.indices((nx,ny))
    rx = np.ravel(x)
    ry = np.ravel(y)
    C = Cmatrix2d(rx,ry,smoothing,smoothing,0)
    B = Bmatrix(C)

    def residual(pars,x,y,data=None):
        model = two_d_gaussian(x,y, pars['amp'].value,pars['xo'].value, pars['yo'].value,
                               pars['major'].value,pars['minor'].value,pars['pa'].value)
        model = np.ravel(model)
        if data is None:
            return model
        resid = (model-data).dot(B)
        return resid

    def residual_nocorr(pars,x,y,data=None):
        model = two_d_gaussian(x,y, pars['amp'].value,pars['xo'].value, pars['yo'].value,
                               pars['major'].value,pars['minor'].value,pars['pa'].value)
        model = np.ravel(model)
        if data is None:
            return model
        resid = (model-data)
        return resid

    # Create the signal
    s_params = lmfit.Parameters()
    s_params.add('amp', value=10.0)
    s_params.add('xo', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    s_params.add('yo', value=1.0*ny/2, min=0.8*ny/2, max=1.2*ny/2)
    s_params.add('major', value=2.*smoothing, min=1.8*smoothing, max=2.2*smoothing)
    s_params.add('minor', value=1.*smoothing, min=0.8*smoothing, max=1.2*smoothing)
    s_params.add('pa', value=0.2, min=-1.*np.pi, max=np.pi)

    signal = residual(s_params, x, y) # returns model as a vector
    signal = signal.reshape(nx,ny)

    # create the initial guess
    iparams = copy.deepcopy(s_params)
    iparams['amp'].value*=1.1
    iparams['xo'].value-=1
    iparams['yo'].value+=1
    iparams['major'].value*=1.05
    iparams['minor'].value*=0.95

    diffs_corr = []
    errs_corr = []
    diffs_nocorr = []
    errs_nocorr = []
    crb_corr = []

    for n in xrange(3):

        np.random.seed(23423 + n)
        noise = np.random.random((nx,ny))*0
        #noise = gaussian_filter(noise, sigma=[smoothing,smoothing])
        #noise -= np.mean(noise)
        #noise /= np.std(noise)

        data = signal + noise
        #mask the data
        #data[data<4] = np.nan
        data, mask, shape = ravel_nans(data)
        x_mask = x[mask]
        y_mask = y[mask]
        if len(x_mask) <6:
            print n, len(x_mask), '(skip)'
            continue
        else:
            print n,len(x_mask)
        C = Cmatrix2d(x_mask,y_mask,smoothing,smoothing,0)
        B = Bmatrix(C)

        pars_corr = copy.deepcopy(iparams)
        mi_corr = lmfit.minimize(residual, pars_corr, args=(x_mask,y_mask, data))#,Dfun=jacobian2d)
        pars_nocorr = copy.deepcopy(iparams)
        mi_nocorr = lmfit.minimize(residual_nocorr, pars_nocorr, args=(x_mask,y_mask, data))#,Dfun=jacobian2d)


        if np.all( [pars_corr[i].stderr >0 for i in pars_corr.valuesdict().keys()]):
            diffs_corr.append([ s_params[i].value -pars_corr[i].value for i in pars_corr.valuesdict().keys()])
            errs_corr.append( [pars_corr[i].stderr for i in pars_corr.valuesdict().keys()])
            #crb_corr.append([ 0 for i in pars_corr.valuesdict().keys()])
            crb_corr.append( CRB_errs(jacobian2d(s_params,x_mask,y_mask),C))
        #print mi_corr.nfev, mi_nocorr.nfev
        #print_par(pars_corr)

        if np.all( [pars_nocorr[i].stderr >0 for i in pars_nocorr.valuesdict().keys()]):
            diffs_nocorr.append([ s_params[i].value -pars_nocorr[i].value for i in pars_nocorr.valuesdict().keys()])
            errs_nocorr.append( [pars_nocorr[i].stderr for i in pars_nocorr.valuesdict().keys()])

    diffs_corr = np.array(diffs_corr)
    errs_corr = np.array(errs_corr)
    diffs_nocorr = np.array(diffs_nocorr)
    errs_nocorr = np.array(errs_nocorr)
    crb_corr = np.array(crb_corr)

    many = n>10
    from matplotlib import pyplot
    if many:
        fig = pyplot.figure(1)
        ax = fig.add_subplot(121)
        ax.plot(diffs_corr[:,0], label='amp')
        ax.plot(diffs_corr[:,1], label='xo')
        ax.plot(diffs_corr[:,2], label='yo')
        ax.plot(diffs_corr[:,3], label='major')
        ax.plot(diffs_corr[:,4], label='minor')
        ax.plot(diffs_corr[:,5], label='pa')
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(diffs_nocorr[:,0], label='amp')
        ax.plot(diffs_nocorr[:,1], label='xo')
        ax.plot(diffs_nocorr[:,2], label='yo')
        ax.plot(diffs_nocorr[:,3], label='major')
        ax.plot(diffs_nocorr[:,4], label='minor')
        ax.plot(diffs_nocorr[:,5], label='pa')
        ax.legend()

    if not many:

        model_nocor = two_d_gaussian(x,y, pars_nocorr['amp'].value,pars_nocorr['xo'].value, pars_nocorr['yo'].value,
                               pars_nocorr['major'].value,pars_nocorr['minor'].value,pars_nocorr['pa'].value)

        model = two_d_gaussian(x,y, pars_corr['amp'].value,pars_corr['xo'].value, pars_corr['yo'].value,
                               pars_corr['major'].value,pars_corr['minor'].value,pars_corr['pa'].value)

        data = signal+noise #unravel_nans(data,mask,shape)
        kwargs = {'interpolation':'nearest','cmap':pyplot.cm.cubehelix,'vmin':-1,'vmax':10, 'origin':'lower'}
        fig = pyplot.figure(2,figsize=(6,12))

        ax = fig.add_subplot(2,1,1)
        cb = ax.imshow(data,**kwargs)
        ax.set_title('data')

        cax = pyplot.colorbar(cb)
        cax.set_label("Flux")
        cax.set_ticks(range(-1,11))

        ax = fig.add_subplot(4,3,7)
        ax.imshow(signal, **kwargs)
        ax.set_title('signal')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,10)
        ax.imshow(noise,**kwargs)
        ax.set_title('noise')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,8)
        ax.imshow(model,**kwargs)
        ax.set_title('model')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,11)
        ax.imshow(data-model,**kwargs)
        ax.set_title('data-model')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,9)
        ax.imshow(model_nocor,**kwargs)
        ax.set_title('model_nocorr')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,12)
        ax.imshow(data-model_nocor,**kwargs)
        ax.set_title('data-model_nocorr')
        rmlabels(ax)

        print "true"
        print_par(s_params)
        print "+corr"
        print_par(pars_corr)
        print "-corr"
        print_par(pars_nocorr)

    if False:
        fig = pyplot.figure(3)
        C = np.vstack( [ gaussian(x,1., i, 1.*smoothing) for i in x ])
        #C[C<1e-3]=0.0
        ax = fig.add_subplot(121)
        cb = ax.imshow(C,interpolation='nearest',cmap=pyplot.cm.cubehelix)
        pyplot.colorbar(cb)

        L,Q = eigh(C)
        print L
        S = np.diag(1/np.sqrt(abs(L)))
        B = Q.dot(S)

        ax = fig.add_subplot(122)
        cb = ax.imshow(B,interpolation='nearest',cmap=pyplot.cm.cubehelix)
        pyplot.colorbar(cb)

    if many:
        print " -- corr --"
        for i,val in enumerate(pars_corr.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}, mean(crb_err) {4}".format(val,np.mean(diffs_corr[:,i]),np.std(diffs_corr[:,i]), np.mean(errs_corr[:,i]),np.mean(crb_corr[:,i]))
        print "-- no corr --"
        for i,val in enumerate(pars_nocorr.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}".format(val,np.mean(diffs_nocorr[:,i]), np.std(diffs_nocorr[:,i]), np.mean(errs_nocorr[:,i]))
    pyplot.show()


def test_lm2d_errs_xyswap():
    """
    :return:
    """

    nhoriz = 24
    nvert = 15
    smoothing = 2 #2./(2*np.sqrt(2*np.log(2))) #5 pixels per beam

    y, x = np.indices((nvert,nhoriz))
    rx = np.ravel(x)
    ry = np.ravel(y)
    C = Cmatrix2d(ry,rx,smoothing,smoothing,0)
    B = Bmatrix(C)

    def residual(pars,y,x,data=None):
        model = two_d_gaussian(y, x, pars['amp'].value,pars['yo'].value, pars['xo'].value,
                               pars['major'].value,pars['minor'].value,pars['pa'].value)
        if data is None:
            return model
        resid = (model-data).dot(B)
        return resid

    def residual_nocorr(pars,y,x,data=None):
        model = two_d_gaussian(y, x, pars['amp'].value,pars['yo'].value, pars['xo'].value,
                               pars['major'].value,pars['minor'].value,pars['pa'].value)
        if data is None:
            return model
        resid = (model-data)
        return resid

    # Create the signal
    s_params = lmfit.Parameters()
    s_params.add('amp', value=10.0)
    s_params.add('xo', value=1.1*nhoriz/2, min=0.8*nhoriz/2, max=1.3*nhoriz/2)
    s_params.add('yo', value=0.9*nvert/2, min=0.8*nvert/2, max=1.3*nvert/2)
    s_params.add('major', value=2.*smoothing, min=1.8*smoothing, max=2.2*smoothing)
    s_params.add('minor', value=1.*smoothing, min=0.8*smoothing, max=1.2*smoothing)
    s_params.add('pa', value=np.pi/6, min=-1.*np.pi, max=np.pi)

    signal = residual(s_params, y, x) # returns model as a vector
    signal = signal.reshape(nvert,nhoriz)

    # create the initial guess
    iparams = copy.deepcopy(s_params)
    iparams['amp'].value*=1.1
    iparams['xo'].value-=1
    iparams['yo'].value+=1
    iparams['major'].value*=1.05
    iparams['minor'].value*=0.95

    diffs_corr = []
    errs_corr = []
    diffs_nocorr = []
    errs_nocorr = []
    crb_corr = []

    for n in xrange(50):

        np.random.seed(23423 + n)
        noise = np.random.random((nvert,nhoriz))
        noise = gaussian_filter(noise, sigma=[smoothing,smoothing])
        noise -= np.mean(noise)
        noise /= np.std(noise)*2

        data = signal + noise
        #mask the data
        # data[data<4] = np.nan
        data, mask, shape = ravel_nans(data)
        x_mask = x[mask]
        y_mask = y[mask]
        if len(x_mask) <6:
            print n, len(x_mask), '(skip)'
            continue
        else:
            print n,len(x_mask)
        C = Cmatrix2d(x_mask,y_mask,smoothing,smoothing,0)
        B = Bmatrix(C)

        pars_corr = copy.deepcopy(iparams)
        mi_corr = lmfit.minimize(residual, pars_corr, args=(y_mask,x_mask, data))#,Dfun=jacobian2d)

        pars_nocorr = copy.deepcopy(iparams)
        mi_nocorr = lmfit.minimize(residual_nocorr, pars_nocorr, args=(y_mask,x_mask, data))#,Dfun=jacobian2d)


        if np.all( [pars_corr[i].stderr >0 for i in pars_corr.valuesdict().keys()]):
            if pars_corr['minor'].value>pars_corr['major'].value:
                pars_corr['minor'],pars_corr['major'] =pars_corr['major'],pars_corr['minor']
                pars_corr['pa']+= np.pi/2
            diffs_corr.append([ s_params[i].value -pars_corr[i].value for i in pars_corr.valuesdict().keys()])
            errs_corr.append( [pars_corr[i].stderr for i in pars_corr.valuesdict().keys()])
            #crb_corr.append([ 0 for i in pars_corr.valuesdict().keys()])
            crb_corr.append( CRB_errs(jacobian2d(s_params,x_mask,y_mask),C))
        #print mi_corr.nfev, mi_nocorr.nfev
        #print_par(pars_corr)

        if np.all( [pars_nocorr[i].stderr >0 for i in pars_nocorr.valuesdict().keys()]):
            if pars_nocorr['minor'].value>pars_nocorr['major'].value:
                pars_nocorr['minor'],pars_nocorr['major'] =pars_nocorr['major'],pars_nocorr['minor']
                pars_nocorr['pa']+= np.pi/2
            diffs_nocorr.append([ s_params[i].value -pars_nocorr[i].value for i in pars_nocorr.valuesdict().keys()])
            errs_nocorr.append( [pars_nocorr[i].stderr for i in pars_nocorr.valuesdict().keys()])

    diffs_corr = np.array(diffs_corr)
    errs_corr = np.array(errs_corr)
    diffs_nocorr = np.array(diffs_nocorr)
    errs_nocorr = np.array(errs_nocorr)
    crb_corr = np.array(crb_corr)

    many = n>10
    from matplotlib import pyplot
    if many:
        fig = pyplot.figure(1)
        ax = fig.add_subplot(121)
        ax.plot(diffs_corr[:,0], label='amp')
        ax.plot(diffs_corr[:,1], label='xo')
        ax.plot(diffs_corr[:,2], label='yo')
        ax.plot(diffs_corr[:,3], label='major')
        ax.plot(diffs_corr[:,4], label='minor')
        ax.plot(diffs_corr[:,5], label='pa')
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(diffs_nocorr[:,0], label='amp')
        ax.plot(diffs_nocorr[:,1], label='xo')
        ax.plot(diffs_nocorr[:,2], label='yo')
        ax.plot(diffs_nocorr[:,3], label='major')
        ax.plot(diffs_nocorr[:,4], label='minor')
        ax.plot(diffs_nocorr[:,5], label='pa')
        ax.legend()

    if not many:

        model_nocor = residual(pars_nocorr, y, x)
        model_nocor = model_nocor.reshape(nvert,nhoriz)
        model = residual(pars_corr,y,x)
        model = model.reshape(nvert,nhoriz)

        data = signal+noise #unravel_nans(data,mask,shape)
        kwargs = {'interpolation':'nearest','cmap':pyplot.cm.cubehelix,'vmin':-1,'vmax':10, 'origin':'lower'}
        fig = pyplot.figure(2,figsize=(6,12))

        ax = fig.add_subplot(2,1,1)
        cb = ax.imshow(data,**kwargs)
        ax.set_title('data')

        cax = pyplot.colorbar(cb)
        cax.set_label("Flux")
        cax.set_ticks(range(-1,11))

        ax = fig.add_subplot(4,3,7)
        ax.imshow(signal, **kwargs)
        ax.set_title('signal')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,10)
        ax.imshow(noise,**kwargs)
        ax.set_title('noise')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,8)
        ax.imshow(model,**kwargs)
        ax.set_title('model')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,11)
        ax.imshow(data-model,**kwargs)
        ax.set_title('data-model')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,9)
        ax.imshow(model_nocor,**kwargs)
        ax.set_title('model_nocorr')
        rmlabels(ax)

        ax = fig.add_subplot(4,3,12)
        ax.imshow(data-model_nocor,**kwargs)
        ax.set_title('data-model_nocorr')
        rmlabels(ax)

        print "true"
        print_par(s_params)
        print "+corr"
        print_par(pars_corr)
        print "-corr"
        print_par(pars_nocorr)

    if False:
        fig = pyplot.figure(3)
        C = np.vstack( [ gaussian(x,1., i, 1.*smoothing) for i in x ])
        #C[C<1e-3]=0.0
        ax = fig.add_subplot(121)
        cb = ax.imshow(C,interpolation='nearest',cmap=pyplot.cm.cubehelix)
        pyplot.colorbar(cb)

        L,Q = eigh(C)
        print L
        S = np.diag(1/np.sqrt(abs(L)))
        B = Q.dot(S)

        ax = fig.add_subplot(122)
        cb = ax.imshow(B,interpolation='nearest',cmap=pyplot.cm.cubehelix)
        pyplot.colorbar(cb)

    if many:
        print " -- corr --"
        for i,val in enumerate(pars_corr.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}, mean(crb_err) {4}".format(val,np.mean(diffs_corr[:,i]),np.std(diffs_corr[:,i]), np.mean(errs_corr[:,i]),np.mean(crb_corr[:,i]))
        print "-- no corr --"
        for i,val in enumerate(pars_nocorr.valuesdict().keys()):
            print "{0}: diff {1:6.4f}+/-{2:6.4f}, mean(err) {3}".format(val,np.mean(diffs_nocorr[:,i]), np.std(diffs_nocorr[:,i]), np.mean(errs_nocorr[:,i]))
    pyplot.show()


def make_data():
    nhoriz = 14
    nvert = 10
    smoothing = 2
    y, x = np.indices((nvert,nhoriz))

    def residual(pars,y,x,data=None):
        model = two_d_gaussian(y, x, pars['amp'].value,pars['yo'].value, pars['xo'].value,
                               pars['major'].value,pars['minor'].value,pars['pa'].value)
        if data is None:
            return model
        resid = (model-data).dot(B)
        return resid

    s_params = lmfit.Parameters()
    s_params.add('amp', value=10.0)
    s_params.add('xo', value=1.0*nhoriz/2+0.2, min=0.8*nhoriz/2, max=1.2*nhoriz/2)
    s_params.add('yo', value=1.0*nvert/2-0.1, min=0.8*nvert/2, max=1.2*nvert/2)
    s_params.add('major', value=2.*smoothing, min=1.8*smoothing, max=2.2*smoothing)
    s_params.add('minor', value=1.*smoothing, min=0.8*smoothing, max=1.2*smoothing)
    s_params.add('pa', value=np.pi/3, min=-1.*np.pi, max=np.pi)

    signal = residual(s_params, y, x)

    import astropy.io.fits as fits
    template = fits.open('Test/Images/1904-66_SIN.fits')
    template[0].data = signal
    template.writeto('test.fits', clobber=True)
    template[0].data*=0
    template.writeto('test_bkg.fits', clobber=True)
    template[0].data+=1
    template.writeto('test_rms.fits', clobber=True)


if __name__ == '__main__':
    # test1d()
    # test2d()
    # test2d_load()
    # test_CRB()
    # JC_err_comp()
    # test_cmatrix()
    # test_jacobian()
    test_two_components()