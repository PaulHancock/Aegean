#! /usr/bin/env python
import numpy as np
import sys, os
from optparse import OptionParser

from scipy.interpolate import griddata
from astropy.io import fits as pyfits

from AegeanTools.running_percentile import RunningPercentiles as RP

def running_filter(data,step_size,box_size):
	"""
	Apply a filter that efficiently calculates a background and rms
	over an image by first calculating said properties on a sparse grid 
	and then interpolating these values over the entire image.

	The background is the median of the surrounding pixels.
	The rms is the inter qartile range of the surrounding pixels (scaled by 1.34896).
	These choices reduce the bias of source pixels in the surrounding region.

	The calculation of bkg/rms is done using a running percentiles filter.
	Input:
	data = a 2d np.array() of image pixels
	step_size = [x,y] (ints) the spacing of the grid points at which the 
	            bkg and rms are calculated
	box_size = [x,y] (ints) the size of the box over which the bkg/rms are
	            calculated
	Returns:
	bkg = np.array() representing the background of the image (same dimensions)
	rms = np.array() representing the rms of the image (same dimesions)
	      both the bkg and rms images will be masked so that pixels that are np.NaN
	      in the input image are also np.NaN in the bkg/rms images
	"""
	#start a new RunningPercentile class
	rp = RP()

	#setup the bkg/rms images
	filtered_bkg = np.zeros(data.shape)
	filtered_rms = np.zeros(data.shape)

	def locations(data,step_size):
		"""
		Generator function to iterate over a grid of x,y coords
		Returns:
		x,y,previous_x,previous_y
		"""
		#initial data
		x,y=0,0 #locations
		px,py=0,0 #previous locations
		yield x,y,px,py #first box
		sizex,sizey=step_size
		shape=data.shape #size of our array
		while True:
			#this needs to move in a zig/zag pattern to 
			#maximise overlaps between adjacent steps
			x+=sizex
			if x>=shape[0] or x<0:
				x-=sizex #go 'back' a step
				y+=sizey
				sizex*=-1 #change our step direction
				if y>=shape[1]:
					break # we are at then end of our iterations
			yield x,y,px,py
			px=x
			py=y

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
	x_min,x_max,y_min,y_max = box(0,0)
	#print "initial box is",x_min,x_max,y_min,y_max
	new = data[x_min:x_max,y_min:y_max].ravel()
	#print "and has",len(new),"pixels"
	rp.add(new)
	for x,y,px,py in locations(data,step_size):
		x_min,x_max,y_min,y_max = box(x,y)
		px_min,px_max,py_min,py_max = box(px,py)
		old=[]
		new=[]
		#print "box is",x_min,x_max,y_min,y_max
		#print data[x_min:x_max,y_min:y_max]

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


	(gx,gy) = np.mgrid[0:data.shape[0],0:data.shape[1]]
	interpolated_bkg = griddata(bkg_points,bkg_values,(gx,gy),method='linear')
	interpolated_rms = griddata(rms_points,rms_values,(gx,gy),method='linear')
	mask = np.where(np.isnan(data))
	interpolated_bkg[mask]=np.NaN
	interpolated_rms[mask]=np.NaN
	#filtered_bkg[mask]=np.NaN
	#filtered_rms[mask]=np.NaN
	return interpolated_bkg,interpolated_rms

def filter_image(im_name,out_base,step_size=None,box_size=None,twopass=False):
	"""

	"""
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

	print "using step_size {0}, box_size {1}".format(step_size,box_size)
	print "on data shape {0}".format(data.shape)
	bkg,rms = running_filter(data,step_size=step_size,box_size=box_size)
	if twopass:
		print "running second pass to get a better rms"
		_,rms=running_filter(data-bkg,step_size=step_size,box_size=box_size)
	bkg_out = '_'.join([os.path.expanduser(out_base),'bkg.fits'])
	rms_out = '_'.join([os.path.expanduser(out_base),'rms.fits'])

	save_image(fits,bkg,bkg_out)
	save_image(fits,rms,rms_out)
	# fits.data = rms
	# fits.writeto(rms_out,clobber=True)
	# print "wrote {0}".format(rms_out)
	# fits.data = bkg
	# fits.writeto(bkg_out,clobber=True)
	# print "wrote {0}".format(bkg_out)
	# return

def load_image(im_name):
	"""
	Generic helper function to load a fits file
	"""
	fits = pyfits.open(im_name)
	data = fits[0].data	
	if fits[0].header['NAXIS']>2:
		data = data.squeeze() #remove axes with dimension 1
	print "loaded {0}".format(im_name)
	return fits,data

def save_image(fits,data,im_name):
	"""
	Generic helper function to save a fits file with a given name/header
	This function modifies the fits object!
	"""
	fits[0].data = data
	fits.writeto(im_name,clobber=True)
	print "wrote {0}".format(im_name)
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
    parser.add_option('--twopass',dest='twopass',action='store_true',
    	              help='Calculate the bkg and rms in a two passes instead of one. (when the bkg changes rapidly)')
    parser.set_defaults(out_base='out',step_size=None,box_size=None,twopass=False)
    (options, args) = parser.parse_args()

    if len(args)<1:
    	parser.print_help()
    	sys.exit()
    else:
    	filename = args[0]
    if not os.path.exists(filename):
        print "{0} does not exist".format(filename)
        sys.exit()

    filter_image(im_name=filename,out_base=options.out_base,step_size=options.step_size,box_size=options.box_size,twopass=options.twopass)
    print "Done"