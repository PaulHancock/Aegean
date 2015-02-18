#! /usr/bin/env python

import numpy as np
import lmfit
import math
import sys
from AegeanTools.mpfit import mpfit
import logging
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

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

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)


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

def test_lm_corr_noise():
    """
    :return:
    """
    nx = 20
    smoothing = 3
    y = gaussian(np.arange(nx), 1, nx/2, smoothing)
    y = np.roll(y,nx/2)
    C = np.vstack((y,)*nx)
    for i in range(nx):
        C[i] = np.roll(C[i],i)

    Ci = np.matrix(C).I
    #Ci = np.matrix(np.diag(np.ones(nx)))
    def residual(pars,x,data=None):
        amp = pars['amp'].value
        cen = pars['cen'].value
        wid = pars['wid'].value
        model = gaussian(x, amp, cen, wid)
        if data is None:
            return model
        resid = (model-data) * Ci # * np.matrix(model-data).T
        return resid.tolist()[0]

    x = np.arange(nx)

    params = lmfit.Parameters()
    params.add('amp', value=10.0, min=9, max=11)
    params.add('cen', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('wid', value=2.0*smoothing, min=smoothing, max=3.0*smoothing)

    signal = gaussian(x, params['amp'].value, params['cen'].value, params['wid'].value)

    np.random.seed(1234567)
    noise = np.random.random(nx)
    noise = gaussian_filter1d(noise, sigma=smoothing)
    noise -= np.mean(noise)
    noise /= np.std(noise)

    data = signal + noise

    mi = lmfit.minimize(residual, params, args=(x,data))
    model = gaussian(x, params['amp'].value, params['cen'].value, params['wid'].value)
    print params
    from matplotlib import pyplot
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,signal, label='signal')
    ax.plot(x,noise, label='noise')
    ax.plot(x,data, label='data')
    ax.plot(x,model, label='model')
    ax.legend()
    pyplot.show()

def test_lm_corr_noise_2d():
    """
    :return:
    """
    nx = 20
    smoothing = 3
    x, y = np.meshgrid(range(nx),range(nx))
    z = np.ravel(two_d_gaussian(x, y, 1, 0, 0, smoothing, smoothing, 0))
    C = np.vstack((z,)*nx*nx)
    for i in range(nx*nx):
        C[i] = np.roll(C[i],i)

    Ci = np.matrix(C).I
    Ci = np.matrix(np.diag(np.ones(nx*nx)))
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
    params.add('amp', value=10.0, min=9, max=11)
    params.add('xo', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('yo', value=1.0*nx/2, min=0.8*nx/2, max=1.2*nx/2)
    params.add('major', value=smoothing, min=0.8*smoothing, max=1.2*smoothing)
    params.add('minor', value=smoothing, min=0.8*smoothing, max=1.2*smoothing)
    params.add('pa', value=0, min=-1.*np.pi, max=np.pi)


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

    kwargs = {'vmin':-1, 'vmax':10, 'interpolation':'nearest'}
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


if __name__ == '__main__':
    # test2d2()
    # compare()
    test_lm_corr_noise_2d()
