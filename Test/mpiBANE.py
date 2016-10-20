#! /usr/bin/env python
from __future__ import print_function
"""
Test of making BANE work with openmpi
"""

__author__ = 'Paul Hancock'
__version__ = 'v0.1'
__date__ = '2016-10-20'

from AegeanTools import BANE
from astropy.io import fits
from mpi4py import MPI
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import sys


def sigma_filter(filename, region, step_size, box_size, shape, dobkg=True):
    """
    Calculated the rms [and background] for a sub region of an image. Save the resulting calculations
    into shared memory - irms [and ibkg].
    :param filename: Fits file to open
    :param region: Region within fits file that is to be processed
    :param step_size: The filtering step size
    :param box_size: The size of the box over which the filter is applied (each step)
    :param shape: The shape of the fits image
    :param dobkg: True = do background calculation.
    :return:
    """

    # Caveat emptor: The code that follows is very difficult to read.
    # xmax is not x_max, and x,y actually should be y,x
    # TODO: fix the code below so that the above comment can be removed

    ymin, ymax, xmin, xmax = region

    cmin = max(0, ymin - box_size[1]//2)
    cmax = min(shape[1], ymax + box_size[1]//2)
    rmin = max(0, xmin - box_size[0]//2)
    rmax = min(shape[0], xmax + box_size[0]//2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]

    # It seems that I cannot memmap the same file multiple times without errors
    with fits.open(filename, memmap=False) as a:
        if NAXIS == 2:
            data = a[0].section[rmin:rmax, cmin:cmax]
        elif NAXIS == 3:
            data = a[0].section[0, rmin:rmax, cmin:cmax]
        elif NAXIS == 4:
            data = a[0].section[0, 0, rmin:rmax, cmin:cmax]
        else:
            sys.exit(1)

    # x/y min/max should refer to indices into data
    # this is the region over which we want to operate
    ymin -= cmin
    ymax -= cmin
    xmin -= rmin
    xmax -= rmin

    def locations(step_size, xmin, xmax, ymin, ymax):
        """
        Generator function to iterate over a grid of x,y coords
        operates only within the given bounds
        Returns:
        x, y
        """

        xvals = range(xmin, xmax, step_size[0])
        if xvals[-1] != xmax:
            xvals.append(xmax)
        yvals = range(ymin, ymax, step_size[1])
        if yvals[-1] != ymax:
            yvals.append(ymax)
        # initial data
        for y in yvals:
            for x in xvals:
                yield x, y

    def box(x, y):
        """
        calculate the boundaries of the box centered at x,y
        with size = box_size
        """
        x_min = max(0, x-box_size[0]//2)
        x_max = min(data.shape[0]-1, x+box_size[0]//2)
        y_min = max(0, y-box_size[1]//2)
        y_max = min(data.shape[1]-1, y+box_size[1]//2)
        return x_min, x_max, y_min, y_max

    bkg_points = []
    bkg_values = []
    rms_points = []
    rms_values = []

    for x, y in locations(step_size, xmin, xmax, ymin, ymax):
        x_min, x_max, y_min, y_max = box(x, y)
        new = data[x_min:x_max, y_min:y_max]
        new = np.ravel(new)
        new = BANE.sigmaclip(new, 3, 3)
        # If we are left with (or started with) no data, then just move on
        if len(new) < 1:
            continue

        if dobkg:
            bkg = np.median(new)
            bkg_points.append((x+rmin, y+cmin))  # these coords need to be indices into the larger array
            bkg_values.append(bkg)
        rms = np.std(new)
        rms_points.append((x+rmin, y+cmin))
        rms_values.append(rms)

    ymin, ymax, xmin, xmax = region
    gx, gy = np.mgrid[xmin:xmax, ymin:ymax]
    # If the bkg/rms calculation above didn't yield any points, then our interpolated values are all nans
    if len(rms_points) > 1:
        # logging.debug("Interpolating rms")
        ifunc = LinearNDInterpolator(rms_points, rms_values)
        # force 32 bit floats
        interpolated_rms = np.array(ifunc((gx, gy)), dtype=np.float32)
        del ifunc
    else:
        interpolated_rms = np.empty((len(gx), len(gy)), dtype=np.float32)*np.nan

    if dobkg:
        if len(bkg_points) > 1:
            # logging.debug("Interpolating bkg")
            ifunc = LinearNDInterpolator(bkg_points, bkg_values)
            interpolated_bkg = np.array(ifunc((gx, gy)), dtype=np.float32)
            del ifunc
        else:
            interpolated_bkg = np.empty((len(gx), len(gy)), dtype=np.float32)*np.nan

    return interpolated_bkg, interpolated_rms

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

image = sys.argv[-1]

data = np.squeeze(fits.open(image)[0].data)

if rank == 0:
    print("BANE master")
    print("Data is ", data.shape)
    # result = np.zeros(data.shape)
else:
    print("BANE worker {0}".format(rank))
    # result = None

ny = data.shape[0]//size
start = rank*ny
end = (rank+1)*ny
if rank == size - 1:
    end = data.shape[0]
bkg, rms = sigma_filter(image, (start, end, 0, data.shape[1]), (15, 15), (90, 90), data.shape)
# data[dslice, :].ravel() + rank

print(bkg.shape)
# root doesn't need to send data to itself
if rank != 0:
    print("sending from {0}".format(rank))
    comm.send(bkg, dest=0)
elif rank == 0:
    print("recieving")
    result = bkg  # root's own section of the data
    for i in range(1, size):  # append all the new sections in order
        result = np.hstack((result, comm.recv(source=i)))
    temp = fits.open(image)
    print(result.shape)
    temp[0].data = result.reshape(data.shape)
    temp[0].data[np.where(np.bitwise_not(np.isfinite(data)))] = np.nan
    temp.writeto('temp_bkg.fits', clobber=True)
    print("wrote temp_bkg.fits")

comm.Barrier()
if rank != 0:
    print("sending from {0}".format(rank))
    comm.send(rms, dest=0)
elif rank == 0:
    print("recieving")
    result = rms  # root's own section of the data
    for i in range(1, size):  # append all the new sections in order
        result = np.hstack((result, comm.recv(source=i)))
    print(result.shape)
    temp[0].data = result.reshape(data.shape)
    temp[0].data[np.where(np.bitwise_not(np.isfinite(data)))] = np.nan
    temp.writeto('temp_rms.fits', clobber=True)
    print("wrote temp_rms.fits")
