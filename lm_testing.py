#! /usr/bin/env python

import numpy as np
import lmfit
import math
import sys
from AegeanTools.mpfit import mpfit
import logging

from scipy.ndimage import label, find_objects
import AegeanTools.flags as flags

def explore(snr,status,queue,bounds,pixel):
    """
    Look for pixels adjacent to <pixel> and add them to the queue
    Don't include pixels that are in the queue
    :param snr: array of bool that is true for pixels we are interested in
    :param status: array that represents the status of pixels
    :param queue: the queue of pixels that we are interested in
    :param bounds: the bounds within which we can explore
    :param pixel: the initial point to start from
    :return: None
    """
    (x, y) = pixel

    if x > 0:
        new = (x - 1, y)
        if snr[new] and (not status[new] & flags.QUEUED):
            queue.append(new)
            status[new] |= flags.QUEUED

    if x < bounds[0]:
        new = (x + 1, y)
        if snr[new] and (not status[new] & flags.QUEUED):
            queue.append(new)
            status[new] |= flags.QUEUED

    if y > 0:
        new = (x, y - 1)
        if snr[new] and (not status[new] & flags.QUEUED):
            queue.append(new)
            status[new] |= flags.QUEUED

    if y < bounds[1]:
        new =  (x, y + 1)
        if snr[new] and (not status[new] & flags.QUEUED):
            queue.append(new)
            status[new] |= flags.QUEUED

def flood(snr,status,bounds,peak):
    """
    Start at pixel=peak and return all the pixels that belong to
    the same blob.
    :param snr: array of bool that is true for pixels we are interested in
    :param status: array that represents the status of pixels
    :param bounds: the bounds within which we can explore
    :param peak: the initial point to start from
    :return: None
    """

    if status[peak] & flags.VISITED:
        return []

    blob = []
    queue = [peak]
    status[peak] |= flags.QUEUED

    for pixel in queue:
        if status[pixel] & flags.VISITED:
            continue

        status[pixel] |= flags.VISITED

        blob.append(pixel)
        explore(snr, status, queue, bounds, pixel)

    return blob

def gen_flood_wrap(data,rmsimg,innerclip,outerclip=None,domask=False):
    """
    <a generator function>
    Find all the sub islands in data.
    Detect islands with innerclip.
    Report islands with outerclip

    type(data) = Island
    return = [(pixels,xmin,ymin)[,(pixels,xmin,ymin)] ]
    where xmin,ymin is the offset of the subisland
    """
    if outerclip is None:
        outerclip=innerclip
    #somehow this avoids problems with multiple cores not working properly!?!?
    #TODO figure out why this is so.
    abspix=abs(data)
        
    status=np.zeros(data.shape,dtype=np.uint8)
    # Selecting PEAKED pixels

    status[np.where(abspix/rmsimg>innerclip)] = flags.PEAKED

    # making pixel list
    ax,ay=np.where(abspix/rmsimg>innerclip)

    #TODO: change this so that I can sort without having to decorate/undecorate
    peaks=[(data[ax[i],ay[i]],ax[i],ay[i]) for i in range(len(ax))]

    if len(peaks) == 0:
        return

    # sorting pixel list - strongest peak should be found first
    peaks.sort(reverse = True)
    if peaks[0][0] < 0:
        peaks.reverse()
    peaks=map(lambda x:x[1:],peaks) #strip the flux data so we are left with just the positions
    bounds=(data.shape[0] - 1, data.shape[1] - 1)

    snr = abspix/rmsimg >= outerclip
    # starting image segmentation
    for peak in peaks:
        blob = flood(snr, status, bounds, peak)
        npix=len(blob)
        if npix>=1:
            xmin = min(blob, key= lambda x:x[0])[0]
            xmax = max(blob, key= lambda x:x[0])[0]
            ymin = min(blob, key= lambda x:x[1])[1]
            ymax = max(blob, key= lambda x:x[1])[1]
            yield data[xmin:xmax+1,ymin:ymax+1], xmin,xmax,ymin,ymax



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
    """

    amp = inpars[::6]
    xo = inpars[1::6]
    yo = inpars[2::6]
    major = inpars[3::6]
    minor = inpars[4::6]
    pa = inpars[5::6]

    #st,ct,s2t=zip(*[ (math.sin(p)**2,math.cos(p)**2,math.sin(2*p)) for p in pa])
    #a, bb, c = zip(*[ ((ct[i]/major[i]**2 + st[i]/minor[i]**2)/2,
    #                   s2t[i]/4 *(1/minor[i]**2-1/major[i]**2),
    #                   (st[i]/major[i]**2 + ct[i]/minor[i]**2)/2) for i in xrange(len(amp))])

    def rfunc(x,y):
        ans=0
        #list comprehension just breaks here, something to do with scope i think
        for i in range(len(amp)):
            ans+= two_d_gaussian(y,x,amp[i], xo[i], yo[i], major[i], minor[i], pa[i])
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

#@profile
def test_lm(data,x,y,init_pars):

    # convert 2d data into 1d lists, masking out nans in the process
    data, mask, shape = ravel_nans(data)
    x = np.ravel(x[mask])
    y = np.ravel(y[mask])

    # setup the fitting
    g1 = lmfit.Model(two_d_gaussian,independent_vars=['x','y'],prefix="c1_") 
    g1.set_param_hint('amp',value=init_pars[0])
    g1.set_param_hint('xo',value=init_pars[1]) 
    g1.set_param_hint('yo',value=init_pars[2])
    g1.set_param_hint('major', value=init_pars[3])
    g1.set_param_hint('minor', value=init_pars[4])
    g1.set_param_hint('pa', value = init_pars[5] , min=-math.pi, max=math.pi)

    g2 = lmfit.Model(two_d_gaussian,independent_vars=['x','y'],prefix="c2_") 
    g2.set_param_hint('amp',value=init_pars[6])
    g2.set_param_hint('xo',value=init_pars[7]) 
    g2.set_param_hint('yo',value=init_pars[8])
    g2.set_param_hint('major', value=init_pars[9])
    g2.set_param_hint('minor', value=init_pars[10])
    g2.set_param_hint('pa', value =init_pars[11] , min=-math.pi, max=math.pi)

    #do the fit
    gmod = reduce(lambda x,y: x+y,[g1+g2])
    params = gmod.make_params()
    result = gmod.fit(data,x=x,y=y,params=params)
    keylist = [a+b for a in ['c1_','c2_'] for b in ['amp','xo','yo','major','minor','pa']]
    return unravel_nans(result.best_fit,mask,shape), [result.values[k] for k in keylist]

#@profile
def test_mpfit(data,init_pars):
    i=1
    parinfo=[]
    parinfo.append( {'value':init_pars[0],
       'fixed':False,
       'parname':'{0}:amp'.format(i),
       'limits':[init_pars[0]*0.8, init_pars[0]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[1],
       'fixed':False,
       'parname':'{0}:xo'.format(i),
       'limits':[init_pars[1]*0.8,init_pars[1]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[2],
       'fixed':False,
       'parname':'{0}:yo'.format(i),
       'limits':[init_pars[2]*0.8,init_pars[2]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[3],
       'fixed': False,
       'parname':'{0}:major'.format(i),
       'limits':[init_pars[3]*0.8,init_pars[3]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[4],
       'fixed': False,
       'parname':'{0}:minor'.format(i),
       'limits':[init_pars[4]*0.8,init_pars[4]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[5],
       'fixed': False,
       'parname':'{0}:pa'.format(i),
       'limits':[-180,180],
       'limited':[False,False]} )

    i=2
    parinfo.append( {'value':init_pars[6],
       'fixed':False,
       'parname':'{0}:amp'.format(i),
       'limits':[init_pars[6]*0.8,init_pars[6]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[7],
       'fixed':False,
       'parname':'{0}:xo'.format(i),
       'limits':[init_pars[7]*0.8,init_pars[7]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[8],
       'fixed':False,
       'parname':'{0}:yo'.format(i),
       'limits':[init_pars[8]*0.8,init_pars[8]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[9],
       'fixed': False,
       'parname':'{0}:major'.format(i),
       'limits':[init_pars[9]*0.8,init_pars[9]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[10],
       'fixed': False,
       'parname':'{0}:minor'.format(i),
       'limits':[init_pars[10]*0.8,init_pars[10]*1.2],
       'limited':[True,True]} )

    parinfo.append( {'value':init_pars[11],
       'fixed': False,
       'parname':'{0}:pa'.format(i),
       'limits':[-180,180],
       'limited':[False,False]} )

    #print parinfo
    mp, parinfo = multi_gauss(data,parinfo)

    ret = ntwodgaussian(mp.params)(*np.indices(data.shape))
    return ret, mp.params


def compare():
    dxy = 40
    dim = range(dxy)
    x,y = np.meshgrid(dim,dim)

    t_pars = [ 1,20,20,6,3,0,   1,10,30,3,3,0]
    init_pars = t_pars *(1+0.2*(0.5-np.random.random(len(t_pars))))

    ztrue = two_d_gaussian(x,y, *t_pars[:6]) + two_d_gaussian(x,y, *t_pars[6:])
    z = ztrue + 0.1*(0.5-np.random.random((dxy,dxy)))
    #z[:10,13:14]=np.nan

    #z=ztrue

    lm_data, lm_pars = test_lm(z,x,y,init_pars)
    mp_data, mp_pars = test_mpfit(z,init_pars)

    
    s_pars = ['amp','xo','yo','maj','min','pa']*2

    print "results:"
    print "     input  init    lm    mp"
    for l,m,t,s,i in zip(lm_pars,mp_pars,t_pars,s_pars,init_pars):
        print "{3:4s} {2:5.2f} {4:5.2f} {0:5.2f} {1:5.2f}".format(l,m,t,s,i)

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

def prof_comp():
    dxy = 25
    dim = range(dxy)
    x,y = np.meshgrid(dim,dim)

    t_pars = [ 1,10,10,3,1,0,   1,13,17,1,1,0]
    init_pars = t_pars +0.01*np.random.random(len(t_pars))

    ztrue = two_d_gaussian(x,y, *t_pars[:6]) + two_d_gaussian(x,y, *t_pars[6:])
    z = ztrue + 0.1*(0.5-np.random.random((dxy,dxy)))
    z[:10,13:14]=np.nan

    #z=ztrue
    for i in xrange(10):
        init_pars = t_pars+0.01*np.random.random(len(t_pars))
        lm_data, lm_pars = test_lm(z,x,y,init_pars)
        mp_data, mp_pars = test_mpfit(z,init_pars)

def scipy_segment(data, rmsimg, innerclip,outerclip=None,domask=False):
    """
    """
    if outerclip is None:
        outerclip = innerclip

    snr = data/rmsimg
    a =  snr > outerclip
    l, n = label(a)
    f = find_objects(l)

    for i in range(n):
        xmin,xmax = f[i][0].start, f[i][0].stop
        ymin,ymax = f[i][1].start, f[i][1].stop
        if np.any(snr[xmin:xmax,ymin:ymax]>innerclip):
            yield data[xmin:xmax,ymin:ymax], xmin, xmax, ymin, ymax

def test_segment():
    """
    """
    a = np.array([[0,1,1,0,0,0],[0,1,1,0,1,0],[0,0,0,1,1,1],[0,0,0,0,1,0]])
    b = np.ones(a.shape)*0.1
    innerclip=5
    print a
    for i,j in zip(scipy_segment(a,b,innerclip),gen_flood_wrap(a,b,innerclip)):
        print i
        print j
        print 

@profile
def prof_segment():
    """
    """
    for i in range(10):
        a = np.random.random((100,100))
        b = np.ones(a.shape)*0.2
        innerclip = 2
        outerclip = 1
        t1 = [ j for j in scipy_segment(a,b,innerclip,outerclip)]
        t2 = [ j for j in gen_flood_wrap(a,b,innerclip,outerclip)]
        print len(t1), len(t2)

if __name__ == '__main__':
    # test2d2()
    # compare()
    # prof_comp()
    # test_segment()
    prof_segment()
