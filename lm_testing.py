#! /usr/bin/env python

import numpy as np
import lmfit
import math
import sys
from AegeanTools.mpfit import mpfit
import logging
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.linalg import sqrtm, eigh, inv
import copy

def unravel_nans(arr,mask,shape):
    """
    Assume that 
    mask = np.where(np.isfinite(data))
    arr = data[mask].ravel()
    shape = data.shape

    Reconstruct an array 'out' such that:
    out.shape == data.shape
    out == data
    np.where(np.isfinite(out)) == np.where(np.isfinite(data))
    """
    #make an array of all nans
    out = np.empty(shape)*np.nan
    #and fill it with the array values according to mask
    out[mask] = arr
    return out

def ravel_nans(arr):
    shape = arr.shape
    mask = np.where(np.isfinite(arr))
    arr1d = np.ravel(arr[mask])
    return arr1d, mask, shape

def two_d_gaussian(x, y, amp, xo, yo, major, minor, pa):
    """
    x,y -> pixels
    xo,yo -> location
    major,minor -> axes in pixels
    pa -> radians
    """
    st, ct, s2t = math.sin(pa)**2,math.cos(pa)**2, math.sin(2*pa)
    a, bb, c = (ct/major**2 + st/minor**2)/2, \
                s2t/4 *(1/minor**2-1/major**2),\
                (st/major**2 + ct/minor**2)/2
    return amp*np.exp(-1*(a*(x-xo)**2 + 2*bb*(x-xo)*(y-yo) + c*(y-yo)**2) )

def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5*((x-cen)/sigma)**2)


def ntwodgaussian(inpars):
    """
    Return an array of values represented by multiple Gaussians as parameterized
    by params = [amp,x0,y0,major,minor,pa]{n}
    x0,y0,major,minor are in pixels
    major/minor are interpreted as being sigmas not FWHMs
    pa is in degrees
    """
    if not len(inpars)%6 ==0:
        logging.error("inpars requires a multiple of 6 parameters")
        logging.error("only {0} parameters supplied".format(len(inpars)))
        sys.exit()
    #pars=np.array(inpars).reshape(len(inpars)/6,6)
    amp,xo,yo,major,minor,pa = np.array(inpars).reshape(6,len(inpars)/6)
    #transform pa->-pa so that our angles are CW instead of CCW
    st,ct,s2t=zip(*[ (math.sin(np.radians(-p))**2,math.cos(np.radians(-p))**2,math.sin(2*np.radians(-p))) for p in pa])
    a, bb, c = zip(*[ ((ct[i]/major[i]**2 + st[i]/minor[i]**2)/2,
                       s2t[i]/4 *(1/minor[i]**2-1/major[i]**2),
                       (st[i]/major[i]**2 + ct[i]/minor[i]**2)/2) for i in xrange(len(amp))])

    def rfunc(x,y):
        ans=0
        #list comprehension just breaks here, something to do with scope i think
        for i in range(len(amp)):
            ans+= amp[i]*np.exp(-1*(a[i]*(x-xo[i])**2 + 2*bb[i]*(x-xo[i])*(y-yo[i]) + c[i]*(y-yo[i])**2) )
        return ans
    return rfunc

def multi_gauss(data,parinfo):
    """
    Fit multiple gaussian components to data using the information provided by parinfo.
    data may contain 'flagged' or 'masked' data with the value of np.NaN
    input: data - pixel information
           parinfo - initial parameters for mpfit
    return: mpfit object, parameter info
    """
    
    data=np.array(data)
    mask=np.where(np.isfinite(data)) #the indices of the *non* NaN values in data

    def model(p):
        """Return a map with a number of gaussians determined by the input parameters."""
        f = ntwodgaussian(p)
        ans = f(*mask)
        return ans

    def erfunc(p,fjac=None):
        """The difference between the model and the data"""
        ans = [0,np.ravel(model(p)-data[mask])]
        return ans
    
    mp=mpfit(erfunc,parinfo=parinfo,quiet=True)

    return mp,parinfo

def Cmatrix(x,sigma):
    return np.vstack( [ gaussian(x,1., i, 1.*sigma) for i in x ])

def Bmatrix(C):
    # this version of finding the square root of the inverse matrix
    # suggested by Cath,
    L,Q = eigh(C)
    # The abs(L) converts negative eigenvalues into positive ones, and stops the B matrix from having nans
    if not all(L>0):
        print "at least one eigenvalue is negative, this will cause problems!"
        sys.exit(-1)
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
    matrix = np.transpose(np.vstack((dmds,dmdcen,dmdsigma)))
    return matrix


def dmdtheta(pars,x,data):
    """
    Calculate the error matrix defined by Cath, but for the 1d gaussian case
    :param pars:
    :param x:
    :return:
    """
    #The matrix is just the jacobian!
    return jacobian(pars,x,data)

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
        fim_inv = inv(np.transpose(jac).dot(inv(C)).dot(jac) )
        # print inv(jac).shape, C.shape, inv(np.transpose(jac)).shape
        # fim_inv = np.linalg.pinv(jac).dot(C).dot(np.linalg.pinv(np.transpose(jac)))

    errs = np.sqrt(np.diag(fim_inv))
    return errs

def test1d():
    x = np.linspace(-5,5,100)
    ytrue = gaussian(x,4,1,2) + gaussian(x,2,-1,2)
    y = ytrue  + 0.4*(0.5-np.random.random(100))

    g1 = lmfit.Model(gaussian,prefix='c1_') 
    g1.set_param_hint('cen',value=0.1,min=0,max=0.8)
    g2 = lmfit.Model(gaussian,prefix='c2_')

    gmod = g1 + g2
    params = gmod.make_params(c1_cen=1, c1_amp=1, c1_wid=1, c2_cen=0, c2_amp=1,c2_wid=1)
    result = gmod.fit(y,x=x,params=params)

    print result.fit_report()
    print dir(result)

    from matplotlib import pyplot
    fig=pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y,'bo')
    ax.plot(x,ytrue,'k--')
    ax.plot(x,result.best_fit,'g-')
    ax.plot(x,gmod.components[0].eval(x=x,params=result.params),'r-')
    ax.plot(x,gmod.components[1].eval(x=x,params=result.params),'b-')
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
    import copy

    nx = 10
    smoothing = 3./(2*np.sqrt(2*np.log(2))) #5 pixels per beam

    x, y = np.meshgrid(range(nx),range(nx))
    C = np.vstack( [ np.ravel(two_d_gaussian(x,y,1, i, j, smoothing, smoothing, 0))
                             for i,j in zip(x.ravel(),y.ravel())])

    # The square root should give a matrix of real values, so the inverse should all be real
    # Some kind of round off effect stops this from being true so we enforce it.
    Ci = abs(np.matrix(sqrtm(C)).I)
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
            resid = (model-data) * Ci  #* np.matrix(model-data).T
            return resid.tolist()[0]

    params = lmfit.Parameters()
    params.add('amp', value=10.0, min=9, max=11)
    params.add('xo', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('yo', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('major', value=smoothing, min=0.8*smoothing, max=1.2*smoothing)
    params.add('minor', value=smoothing, min=0.8*smoothing, max=1.2*smoothing)
    params.add('pa', value=0, min=-1.*np.pi, max=np.pi)

    signal = residual(params, x, y) # returns model
    signal = signal.reshape(nx,nx)

    diffs = []
    errs = []

    for n in xrange(50):
        # need to re-init this.
        print n

        pars = copy.deepcopy(params)

        np.random.seed(1234567 + n)
        noise = np.random.random((nx,nx))
        noise = gaussian_filter(noise, sigma=smoothing)
        noise -= np.mean(noise)
        noise /= np.std(noise)

        data = np.ravel(signal + noise)

        mi = lmfit.minimize(residual, pars, args=(x, y, data))
        if np.all( [pars[i].stderr >0 for i in params.valuesdict().keys()]):
            diffs.append([ params[i].value -pars[i].value for i in params.valuesdict().keys()])
            errs.append( [pars[i].stderr for i in params.valuesdict().keys()])

    diffs = np.array(diffs)
    errs = np.array(errs)
    # print diffs

    # ratios = np.array(ratios)
    # print ratios

    # from matplotlib import pyplot
    # fig = pyplot.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(diffs[:,0], label='amp')
    # ax.plot(diffs[:,1]*10, label='xo')
    # ax.plot(diffs[:,2]*1e2, label='yo')
    # ax.plot(diffs[:,3]*1e3, label='major')
    # ax.plot(diffs[:,4]*1e4, label='minor')
    # ax.set_yscale('log')
    # ax.legend()
    for i,val in enumerate(params.valuesdict().keys()):
        print "{0}: rms(diff) {1}, mean(err) {2}".format(val,np.std(diffs[:,i]), np.mean(errs[:,i]))
    # pyplot.show()

if __name__ == '__main__':
    # test2d2()
    # compare()
    # test_lm_corr_noise()
    # test_lm_corr_noise_2d()
    # test_lm2d_errs()
    test_lm1d_errs()
