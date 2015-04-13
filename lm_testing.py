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
                result += elliptical_gaussian(x,y,amp,xo,yo,sx,sy,theta)
            else:
                result =  elliptical_gaussian(x,y,amp,xo,yo,sx,sy,theta)
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


def do_lmfit(data, params, B=None, D=2):
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

    def residual(params,data=None):
        if D==2:
            f = ntwodgaussian_lmfit(params)
            model = f(*mask)
        elif D==1:
            model = gaussian(mask,params['amp'].value,params['cen'].value,params['sigma'].value)

        if data is None:
            return model
        if B is None:
            return model-data[mask]
        else:
            return (model - data[mask]).dot(B)

    result = lmfit.minimize(residual, params, args=(data,))#,Dfun=jacobian2d)
    return result, params


def Cmatrix(x,sigma):
    return np.vstack( [ gaussian(x,1., i, 1.*sigma) for i in x ])


def Cmatrix2d(x,y,sigmax,sigmay,theta):
    """

    :param x:
    :param y:
    :param sigmax:
    :param sigmay:
    :param theta:
    :return:
    """

    # 1.*sigma avoid stupid integer problems within two_d_gaussian
    f = lambda i,j: np.ravel(elliptical_gaussian(x,y,1,i,j,1.*sigmax,1.*sigmay,theta))
    C = np.vstack( [ f(i,j) for i,j in zip(x,y)] )
    return C


def test_Cmatrix2d():
    nx = 4
    ny = 6
    smoothing = 1.4
    x, y = np.meshgrid(range(nx),range(nx))
    #print [ (x[i,j],y[i,j]) for i in range(len(x)) for j in range(len(y))]
    C = Cmatrix2d(x,y,smoothing,smoothing,0)
    print C


def Bmatrix(C):
    # this version of finding the square root of the inverse matrix
    # suggested by Cath,
    L,Q = eigh(C)
    # The abs(L) converts negative eigenvalues into positive ones, and stops the B matrix from having nans
    if not all(L>0):
        print L
        print "at least one eigenvalue is negative, this will cause problems!"
        sys.exit(1)
    S = np.diag(1/np.sqrt(L))
    B = Q.dot(S)
    return B


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


def jacobian2d(pars,x,y,data=None):
    # print [ k for k in pars.valuesdict().keys()]
    amp = pars['amp'].value
    xo = pars['xo'].value
    yo = pars['yo'].value
    sx = pars['sx'].value
    sy = pars['sy'].value
    theta  = pars['theta'].value
    sint = np.sin(theta)
    cost = np.cos(theta)

    model = two_d_gaussian(x,y, amp,xo, yo, sx, sy, theta)
    model = np.ravel(model)

    dmds = model/amp

    dmdxo = cost * (( x - xo) * cost + (y - yo) * sint) /sy**2
    dmdxo += sint* (( x - xo) * sint - (y - yo) * cost) /sx**2
    dmdxo *= model

    dmdyo = sint * (( x - xo) * cost + (y - yo) * sint) /sy**2
    dmdyo -= cost* (( x - xo) * sint - (y - yo) * cost) /sx**2
    dmdyo *= model

    dmdsy = model / sy**3 * ((x - xo) * cost + (y - yo) * sint)**2
    dmdsx = model / sx**3 * ((x - xo) * sint - (y - yo) * cost)**2

    dmdtheta = -1/sy**2 * ((-1*(x - xo) * sint + (y - yo) * cost) * ((x - xo) * cost + (y - yo) * sint))
    dmdtheta += -1/sx**2 * (((x - xo) * sint - (y - yo) * cost) * ((x - xo) * cost + (y - yo) * sint))
    dmdtheta *= model

    matrix = np.transpose(np.vstack((dmds,dmdxo,dmdyo,dmdsx,dmdsy,dmdtheta)))
    return matrix


def CRB_errs(jac, C, B=None):
    """

    :param jac: the jacobian
    :param C: the correlation matrix
    :param B: B.dot(B') should = inv(C), ie B ~ sqrt(inv(C))
    :return:
    """
    if B is not None:
        # B is actually only a square root of the inverse covariance matrix
        fim_inv =  inv(np.transpose(jac).dot(B).dot(np.transpose(B)).dot(jac))
    else:
        #fim = np.transpose(jac).dot(inv(C)).dot(jac)
        # print_mat(jac)
        # print 'C='
        # print_mat(C)
        # print "inv(C)="
        # print_mat(inv(C))
        # print "fim ="
        # print_mat(fim)
        #fim_inv = inv(fim)
        fim_inv = pinv(jac).dot(C).dot(pinv(np.transpose(jac)))
        # print inv(jac).shape, C.shape, inv(np.transpose(jac)).shape
        # fim_inv = np.linalg.pinv(jac).dot(C).dot(np.linalg.pinv(np.transpose(jac)))

    errs = np.sqrt(np.diag(fim_inv))
    return errs


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
    params = lmfit.Parameters()
    params.add('amp',value=1, min=0.5, max=2)
    params.add('cen',value=1.*nx/2+0.2, min= nx/4., max = 3.*nx/4)
    params.add('sigma', value=1*smoothing, min=0.5*smoothing, max=2*smoothing)

    signal = gaussian(x,params['amp'].value,params['cen'].value,params['sigma'].value)
    snr = 1

    diffs_nocorr = []
    errs_nocorr = []
    diffs_corr = []
    errs_corr = []
    crb_corr = []

    for i in xrange(100):
        np.random.seed(1234567+i)
        noise = np.random.random(nx)
        noise = gaussian_filter1d(noise, sigma=smoothing)
        noise -= np.mean(noise)
        noise /= np.std(noise)*snr

        data = signal+noise
        result,fit_params = do_lmfit(data, params, D=1)
        C = Cmatrix(x,smoothing)
        B = Bmatrix(C)
        result,corr_fit_params = do_lmfit(data, params, D=1, B=B)

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

    many = i>10

    if not many:
        model = gaussian(x, fit_params['amp'].value, fit_params['cen'].value, fit_params['sigma'].value)
        print "init ",
        print_par(params)
        print "final",
        print_par(fit_params)

        from matplotlib import pyplot
        fig=pyplot.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(x,signal,'bo',label='Signal')
        ax.plot(x,noise,'k--',label='Noise')
        ax.plot(x,data,'r-',label="Data")
        ax.plot(x,model,'g-',label="Model")
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
    dxy = 20
    dim = np.linspace(0,5,dxy)
    x,y = np.meshgrid(dim,dim)
    ztrue = two_d_gaussian(x,y, 1, 2.5, 2.5, 3, 1, 0)
    z = ztrue + 0.4*(0.5-np.random.random((dxy,dxy)))
    z[:10,13:14]=np.nan
    
    # convert 2d data into 1d lists, masking out nans in the process
    data, mask, shape = ravel_nans(z)
    x = np.ravel(x[mask])
    y = np.ravel(y[mask])
    # z = np.ravel(z)
    # x = np.ravel(x)
    # y = np.ravel(y)

    # setup the fitting
    g1 = lmfit.Model(two_d_gaussian,independent_vars=['x','y']) 
    g1.set_param_hint('amp',value=1)
    g1.set_param_hint('xo',value=1) 
    g1.set_param_hint('yo',value=1)
    g1.set_param_hint('major', value=2, min=1, max=4)
    g1.set_param_hint('minor', value=1, min=0.5, max=3)
    g1.set_param_hint('pa', value = 0 , min=-math.pi, max=math.pi)

    #do the fit
    gmod = g1 + None
    params = gmod.make_params()
    result = gmod.fit(data,x=x,y=y,params=params)

    print result.fit_report()

    # convert the 1d arrays back into 2d arrays, preserving nans
    z = unravel_nans(data,mask,shape)
    ff = result.best_fit.copy()
    ff = unravel_nans(ff,mask,shape)
    # z = z.reshape((dxy,dxy))
    # ff = ff.reshape((dxy,dxy))

    # plot the three data sets
    from matplotlib import pyplot
    fig=pyplot.figure()
    kwargs = {'interpolation':'nearest','cmap':pyplot.cm.cubehelix}
    ax = fig.add_subplot(2,2,1)
    ax.imshow(ztrue,**kwargs)
    ax.set_title('True')

    ax = fig.add_subplot(2,2,2)
    ax.imshow(z,**kwargs)
    ax.set_title('Data')

    ax = fig.add_subplot(2,2,3)
    ax.imshow(ff,**kwargs)
    ax.set_title('Fit')

    pyplot.show()

def test2d2():
    dxy = 20
    dim = np.linspace(0,5,dxy)
    x,y = np.meshgrid(dim,dim)
    ztrue = two_d_gaussian(x,y, 1, 1, 1, 3, 1, 0) + two_d_gaussian(x,y, 1,3,3,1,1,0)
    z = ztrue + 0.4*(0.5-np.random.random((dxy,dxy)))
    z[:10,13:14]=np.nan
    
    # convert 2d data into 1d lists, masking out nans in the process
    data, mask, shape = ravel_nans(z)
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

    print result.fit_report()
    print params.valuesdict().keys()
    print result.values.keys()

    sys.exit()

    # convert the 1d arrays back into 2d arrays, preserving nans
    z = unravel_nans(data,mask,shape)
    ff = result.best_fit.copy()
    ff = unravel_nans(ff,mask,shape)
    # z = z.reshape((dxy,dxy))
    # ff = ff.reshape((dxy,dxy))

    # plot the three data sets
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
    ax.imshow(ff,**kwargs)
    ax.set_title('Fit')

    ax = fig.add_subplot(2,2,4)
    ax.imshow(ff-ztrue,**kwargs)
    ax.set_title('Fit - True')

    pyplot.show()

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
    test1d()
