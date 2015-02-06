#! /usr/bin/env python

#standard imports
import numpy as np
import sys, os
from optparse import OptionParser
from time import gmtime, strftime
import logging
import copy

#image manipulation 
from scipy.interpolate import griddata
from astropy.io import fits as pyfits

#Aegean tools
from AegeanTools.running_percentile import RunningPercentiles as RP
import AegeanTools.pprocess as pprocess
from AegeanTools.fits_interp import compress

import multiprocessing
#from blist import *

version='v1.0'

###
#
###

class dummy:
    pass

gdata=dummy()
gdata.data = None
gdata.step_size=None
gdata.box_size=None

            
#@profile
def running_filter(xmn,xmx,ymn,ymx):
    """
    A version of running_filter that works on a subset of the data
    and returns the data without interpolation
    uses gdata for: data, step_size, and box_size
    Input:
    bounds - 
    """
    #TODO propagate this change to the remainder of the func
    xmin,xmax,ymin,ymax=ymn,ymx,xmn,xmx
    data=gdata.data
    box_size=gdata.box_size
    step_size=gdata.step_size
    #start a new RunningPercentile class
    rp = RP()
    logging.debug('{0}x{1},{2}x{3} starting at {4}'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    def locations(step_size,xmin,xmax,ymin,ymax):
        """
        Generator function to iterate over a grid of x,y coords
        operates only within the given bounds
        Returns:
        x,y,previous_x,previous_y
        """

        xvals = range(xmin,xmax,step_size[0])
        if xvals[-1]!=xmax:
            xvals.append(xmax)
        yvals = range(ymin,ymax,step_size[1])
        if yvals[-1]!=ymax:
            yvals.append(ymax)
        #initial data
        px,py=xvals[0],yvals[0]
        i=1
        for y in yvals:
            for x in xvals[::i]:
                yield x,y,px,py
                px,py=x,y
            i*=-1 #change x direction

    def box(x,y):
        """
        calculate the boundaries of the box centered at x,y
        with size = box_size
        """
        x_min = max(0,x-box_size[0]/2)
        x_max = min(data.shape[0]-1,x+box_size[0]/2)
        y_min = max(0,y-box_size[1]/2)
        y_max = min(data.shape[1]-1,y+box_size[1]/2)
        return x_min,x_max,y_min,y_max

    bkg_points = []
    rms_points = []
    bkg_values = []
    rms_values = []
    #intialise the rp with our first box worth of data
    x_min,x_max,y_min,y_max = box(xmin,ymin)
    #print "initial box is",x_min,x_max,y_min,y_max
    new = data[x_min:x_max,y_min:y_max].ravel()
    #print "and has",len(new),"pixels"
    rp.add(new)
    for x,y,px,py in locations(step_size,xmin,xmax,ymin,ymax):
        x_min,x_max,y_min,y_max = box(x,y)
        px_min,px_max,py_min,py_max = box(px,py)
        old=[]
        new=[]
        #we only move in one direction at a time, but don't know which
        if (x_min>px_min) or (x_max>px_max):
            #down
            if x_min != px_min:
                old = data[min(px_min,x_min):max(px_min,x_min),y_min:y_max].ravel()
            if x_max != px_max:
                new = data[min(px_max,x_max):max(px_max,x_max),y_min:y_max].ravel()
        elif (x_min<px_min) or (x_max<px_max):
            #up
            if x_min != px_min:
                new = data[min(px_min,x_min):max(px_min,x_min),y_min:y_max].ravel()
            if x_max != px_max:
                old = data[min(px_max,x_max):max(px_max,x_max),y_min:y_max].ravel()
        else: # x's have not changed
            #we are moving right
            if y_min != py_min:
                old = data[x_min:x_max,min(py_min,y_min):max(py_min,y_min)].ravel()
            if y_max != py_max:
                new = data[x_min:x_max,min(py_max,y_max):max(py_max,y_max)].ravel()
        rp.add(new)
        rp.sub(old)
        p0,p25,p50,p75,p100 = rp.score()
        if p50 is not None:
            bkg_points.append((x,y))
            bkg_values.append(p50)
        if (p75 is not None) and (p25 is not None):
            rms_points.append((x,y))
            rms_values.append((p75-p25)/1.34896)
    #return our lists, the interpolation will be done on the master node
    #also tell the master node where the data came from
    logging.debug('{0}x{1},{2}x{3} finished at {4}'.format(xmin,xmax,ymin,ymax,strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    return xmin,xmax,ymin,ymax,bkg_points,bkg_values,rms_points,rms_values

def gen_factors(m,permute=True):
    """
    Generate a list of integer factors for m
    :param m: A positive integer
    :return:
    """
    #convert to int if people have been naughty
    n=int(abs(m))
    #brute force the factors, one of which is always less than sqrt(n)
    for i in xrange(1,int(n**0.5+1)):
        if n%i==0:
            yield i,n/i
            #yield the reverse pair if it is unique
            if i != n/i and permute:
                yield n/i,i

def optimum_sections(cores,data_shape):
    """
    Choose the best sectioning scheme based on the number of cores available and the shape of the data
    :param cores: Number of available cores
    :param data_shape: Shape of the data as [x,y]
    :return: (nx,ny) the number of divisions in each direction
    """
    if cores==1:
        return (1,1)
    if cores%1==1:
        cores-=1
    x,y=data_shape
    min_overlap=np.inf
    best=(1,1)
    for (mx,my) in gen_factors(cores):
        overlap=x*(my-1) + y*(mx-1)
        if overlap<min_overlap:
            best=(mx,my)
            min_overlap=overlap
    logging.debug("Sectioning chosen to be {0[0]}x{0[1]} for a score of {1}".format(best,min_overlap))
    return best

def mask_img(data,mask_data):
    """

    :param data:
    :param mask_data:
    :return:
    """
    mask = np.where(np.isnan(mask_data))
    data[mask]=np.NaN

def filter_mc(data,step_size,box_size,cores):
    """
    Perform a running filter over multiple cores
    """
    #set up the global data for the worker nodes to access
    global gdata
    gdata.data=data
    gdata.step_size=step_size
    gdata.box_size=box_size

    if cores is None:
        cores = multiprocessing.cpu_count()
    if cores>1:
        try:
            queue = pprocess.Queue(limit=cores,reuse=1)
            parfilt = queue.manage(pprocess.MakeReusable(running_filter))
        except AttributeError, e:
            if 'poll' in e.message:
                logging.warn("Your O/S doesn't support select.poll(): Reverting to cores=1")
                cores=1
            else:
                logging.error("Your system can't seem to make a queue, try using --cores=1")
                raise e
    img_y,img_x = data.shape
    if cores>1:
        logging.info("using {0} cores".format(cores))
        nx,ny=optimum_sections(cores,data.shape)

        #box widths should be multiples of the step_size, and not zero
        width_x = max(img_x/nx/step_size[0],1)*step_size[0]
        width_y = max(img_y/ny/step_size[1],1)*step_size[1]
        
        xstart=width_x
        ystart=width_y
        xend=img_x - img_x%width_x #the end point of the last "full" box
        yend=img_y - img_y%width_y
        
        #locations of the box edges
        xmins=[0]
        xmins.extend(range(xstart,xend,width_x))

        xmaxs=[xstart]
        xmaxs.extend(range(xstart+width_x,xend+1,width_x))
        xmaxs[-1]=img_x
        
        ymins=[0]
        ymins.extend(range(ystart,yend,width_y))

        ymaxs=[ystart]
        ymaxs.extend(range(ystart+width_y,yend+1,width_y))
        ymaxs[-1]=img_y
    
        for xmin,xmax in zip(xmins,xmaxs):
            for ymin,ymax in zip(ymins,ymaxs):
                parfilt(xmin,xmax,ymin,ymax)

        #now unpack the results
        bkg_points=[]
        bkg_values=[]
        rms_points=[]
        rms_values=[]
        for xmin,xmax,ymin,ymax,bkg_p,bkg_v,rms_p,rms_v in queue:
            bkg_points.extend(bkg_p)
            bkg_values.extend(bkg_v)
            rms_points.extend(rms_p)
            rms_values.extend(rms_v)
    else:
        #single core we do it all at once
        _,_,_,_,bkg_points,bkg_values,rms_points,rms_values=running_filter(0,img_x,0,img_y)
    #and do the interpolation etc...
    (gx,gy) = np.mgrid[0:data.shape[0],0:data.shape[1]]
    #if the bkg/rms points have len zero this is because they are all nans so we return
    # arrays of nans
    if len(bkg_points)>0:
        interpolated_bkg = griddata(bkg_points,bkg_values,(gx,gy),method='linear')
    else:
        interpolated_bkg=gx*np.nan
    if len(rms_points)>0:
        interpolated_rms = griddata(rms_points,rms_values,(gx,gy),method='linear')
    else:
        interpolated_rms=gx*np.nan

    if cores>1:
        del queue, parfilt
    return interpolated_bkg,interpolated_rms

def filter_image(im_name, out_base, step_size=None, box_size=None, twopass=False, cores=None, mask=True, compressed=False):
    """

    :param im_name:
    :param out_base:
    :param step_size:
    :param box_size:
    :param twopass:
    :param cores:
    :param mask:
    :param compress:
    :return:
    """
    fits,data = load_image(im_name)
    #remove spurious dimensions of this array
    newshape = [a for a in data.shape if a>1]
    if len(newshape) < len(data.shape):
        data = data.copy()
        data.resize(newshape)
    #TODO: if CDELT1 is not found, then look for CD1_1 instead, etc for CDELT2
    if step_size is None:
        if 'BMAJ' in fits[0].header and 'BMIN' in fits[0].header:
            beam_size = np.sqrt(abs(fits[0].header['BMAJ']*fits[0].header['BMIN']))
            if 'CDELT1' in fits[0].header:
                pix_scale = np.sqrt(abs(fits[0].header['CDELT1']*fits[0].header['CDELT2']))
            elif 'CD1_1' in fits[0].header:
                pix_scale = np.sqrt(abs(fits[0].header['CD1_1']*fits[0].header['CD2_2']))
                if fits[0].header['CD1_2'] != 0 or fits[0].header['CD2_1']!=0:
                    logging.warn("CD1_2 and/or CD2_1 are non-zero and I don't know what to do with them")
                    logging.warn("Ingoring them")
            else:
                logging.warn("Cannot determine pixel scale, assuming 4 pixels per beam")
                pix_scale = beam_size/4.
            #default to 4x the synthesized beam width
            step_size = int(np.ceil(4*beam_size/pix_scale))
        else:
            logging.info("BMAJ and/or BMIN not in fits header. Using step_size = 4 pixels")
            step_size = 4
        step_size = (step_size,step_size)

    if box_size is None:
        #default to 5x the step size
        box_size = (step_size[0]*5,step_size[1]*5)

    if compressed:
        if not step_size[0] == step_size[1]:
            step_size = (min(step_size),min(step_size))
            logging.info("Changing grid to be {0} so we can compress the output".format(step_size))

    logging.info("using grid_size {0}, box_size {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(data.shape))
    bkg,rms = filter_mc(data,step_size=step_size,box_size=box_size,cores=cores)
    logging.info("done")
    if twopass:
        logging.info("running second pass to get a better rms")
        _,rms=filter_mc(data-bkg,step_size=step_size,box_size=box_size,cores=cores)
        logging.info("done")

    bkg_out = '_'.join([os.path.expanduser(out_base),'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base),'rms.fits'])

    # force float 32s to avoid bloat
    bkg = np.array(bkg, dtype=np.float32)
    rms = np.array(rms, dtype=np.float32)

    if compressed:
        #need to copy since compress will mess with the header
        old_header= copy.deepcopy(fits[0].header)
        fits[0].data = bkg
        compress(fits, step_size[0], bkg_out)
        fits[0].header = old_header
        fits[0].data = rms
        compress(fits, step_size[0], rms_out)
        return
    if mask:
        mask_img(bkg, data)
        mask_img(rms, data)
    save_image(fits, bkg, bkg_out)
    save_image(fits, rms, rms_out)

###
# Alternate Filters
# Used only for testing algorithm speeds, not really useful
###
def scipy_filter(im_name,out_base,step_size,box_size,cores=None):
    from scipy.ndimage.filters import generic_filter
    from scipy.stats import nanmedian,nanstd,scoreatpercentile

    fits,data = load_image(im_name)

    if step_size is None:
        pix_scale = np.sqrt(abs(fits[0].header['CDELT1']*fits[0].header['CDELT2']))
        beam_size = np.sqrt(abs(fits[0].header['BMAJ']*fits[0].header['BMIN']))
        #default to 4x the synthesized beam width               
        step_size = int(np.ceil(4*beam_size/pix_scale))
        step_size = (step_size,step_size)

    if box_size is None:
        #default to 5x the step size
        box_size = (step_size[0]*5,step_size[1]*5)

    logging.info("using grid {0}, box {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(data.shape))
    logging.info("with scipy generic filter median/std")
    #scipy can't handle nan values when using score at percentile
    def iqrms(x):
        d=x[np.isfinite(x)]
        if len(d)<2:
            return np.nan
        a=scoreatpercentile(d,[75,25])
        return  (a[0]-a[1])/1.34896
    def median(x):
        d=x[np.isfinite(x)]
        if len(d)<2:
            return np.nan
        a=scoreatpercentile(d,50)
        return a

    bkg = generic_filter(data,median,size=box_size)
    rms = generic_filter(data-bkg,iqrms,size=box_size)

    bkg_out = '_'.join([os.path.expanduser(out_base),'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base),'rms.fits'])
    #masking
    mask = np.isnan(data)
    bkg[mask]=np.NaN
    rms[mask]=np.NaN

    save_image(fits,bkg,bkg_out)
    save_image(fits,rms,rms_out)

###
# Helper functions
###
def load_image(im_name):
    """
    Generic helper function to load a fits file
    """
    try:
        fits = pyfits.open(im_name)
    except IOError,e:
        if "END" in e.message:
            logging.warn(e.message)
            logging.warn("trying to ignore this, but you should really fix it")
            fits = pyfits.open(im_name,ignore_missing_end=True)

    data = fits[0].data 
    if fits[0].header['NAXIS']>2:
        data = data.squeeze() #remove axes with dimension 1
    logging.info("loaded {0}".format(im_name))
    return fits,data

def save_image(fits,data,im_name):
    """
    Generic helper function to save a fits file with a given name/header
    This function modifies the fits object!
    """
    fits[0].data = data
    try:
        fits.writeto(im_name,clobber=True)
    except pyfits.verify.VerifyError,e:
        if "DATAMAX" in e.message or "DATAMIN" in e.message:
            logging.warn(e.message)
            logging.warn("I will fix this but it will cause some programs to break")
            fits.writeto(im_name,clobber=True,output_verify="silentfix")
    logging.info("wrote {0}".format(im_name))
    return

#command line version of this program runs from here.    
if __name__=="__main__":
    usage="usage: %prog [options] FileName.fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--out",dest='out_base',
                      help="Basename for output images default = 'out'. eg out_rms.fits, out_bkg.fits")
    parser.add_option('--grid',dest='step_size',type='int',nargs=2,
                      help='The [x,y] size of the grid to use. Default = ~4* beam size square.')
    parser.add_option('--box',dest='box_size',type='int',nargs=2,
                      help='The [x,y] size of the box over which the rms/bkg is calculated. Default = 5*grid.')
    parser.add_option('--cores',dest='cores',type='int',
                      help='Number of cores to use. Default = all available.')
    parser.add_option('--onepass',dest='twopass',action='store_false', help='the opposite of twopass. default=False')
    parser.add_option('--twopass',dest='twopass',action='store_true',
                      help='Calculate the bkg and rms in a two passes instead of one. (when the bkg changes rapidly)')
    parser.add_option('--nomask', dest='mask', action='store_false', default=True,
                      help="Don't mask the output array [default = mask]")
    parser.add_option('--scipy',dest='usescipy',action='store_true',
                      help='Use scipy generic filter instead of the running percentile filter. (for testing/timing)')
    parser.add_option('--debug',dest='debug',action='store_true',help='debug mode, default=False')
    parser.add_option('--compress', dest='compress', action='store_true',default=False,
                      help='Produce a compressed output file.')
    parser.set_defaults(out_base='out',step_size=None,box_size=None,twopass=True,cores=None,usescipy=False,debug=False)
    (options, args) = parser.parse_args()

    logging_level = logging.DEBUG if options.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is BANE {0}".format(version))
    if len(args)<1:
        parser.print_help()
        sys.exit()
    else:
        filename = args[0]
    if not os.path.exists(filename):
        logging.error("{0} does not exist".format(filename))
        sys.exit()
    if options.usescipy:
        scipy_filter(im_name=filename,out_base=options.out_base,step_size=options.step_size,box_size=options.box_size,cores=options.cores)
    else:
        filter_image(im_name=filename, out_base=options.out_base, step_size=options.step_size,
                     box_size=options.box_size, twopass=options.twopass, cores=options.cores,
                     mask=options.mask, compressed=options.compress)

