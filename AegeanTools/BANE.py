#! /usr/bin/env python

"""
This module contains all of the BANE specific code
The function filter_image should be imported from elsewhere and run as is.
"""

from typing import Tuple, Union, TypeAlias, Dict, Any, Optional

import copy
import logging
import multiprocessing
import os
import sys
import uuid
from multiprocessing.shared_memory import SharedMemory
from time import gmtime, strftime

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import normaltest

from .fits_tools import compress 
from .sigma import (
    FittedSigmaClip, fit_bkg_rms_estimate, sigmaclip, fitted_sigma_clip, SigmaClip, FitBkgRmsEstimate
)

__author__ = 'Paul Hancock'
__version__ = 'v1.10.0'
__date__ = '2022-08-17'

BANE_MODE_MAPPINGS = dict(
    sigmaclip=SigmaClip,
    fitrmsbkgestimate=FitBkgRmsEstimate,
    fittedsigmaclip=FittedSigmaClip
)
AVAILABLE_MODES = tuple(BANE_MODE_MAPPINGS.keys())
AVAILABLE_TYPES = tuple(BANE_MODE_MAPPINGS.values())
ClippingModes: TypeAlias = Union[SigmaClip, FitBkgRmsEstimate, FittedSigmaClip]

# global barrier for multiprocessing
barrier = None
memory_id = None

def init(b, mem):
    """
    Set the global barrier and memory_id
    """
    global barrier, memory_id
    barrier = b
    memory_id = mem


def _sf2(args):
    """
    A shallow wrapper for sigma_filter.

    Parameters
    ----------
    args : list
        A list of arguments for sigma_filter

    Returns
    -------
    None
    """
    # an easier to debug traceback when multiprocessing
    # thanks to https://stackoverflow.com/a/16618842/1710603
    try:
        return sigma_filter(*args)
    except Exception as e:
        import traceback
        logging.warn(e)
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def box(r, c, data_shape, box_size):
    """
    calculate the boundaries of the box centered at r,c
    with size = box_size
    """
    r_min = max(0, r - box_size[0] // 2)
    r_max = min(data_shape[0] - 1, r + box_size[0] // 2)
    c_min = max(0, c - box_size[1] // 2)
    c_max = min(data_shape[1] - 1, c + box_size[1] // 2)
    
    return r_min, r_max, c_min, c_max


def adaptive_box_estimate(
    data: np.ndarray, row: int, column: int, box_size, mode: ClippingModes, max_loop: int = 5
) -> Tuple[float,float]:
    """Estimate the background and RMS level within a data extract. Multiple
    clipping and noise estimation modes are supported. 
    
    The adaptive behavour is only activated should the pixel statistics become
    compromised. At the moment, this only is considered true if fewer than 80percent
    of the pixels remain after clipping. 

    Parameters
    ----------
    data : np.ndarray
        Data that will be the source of the box extraction
    row : int
        Row to center extracted region at
    column : int
        Column to center extracted region at
    box_size : Tuple[int,int]
        Base size of the box to extract
    mode : ClippingModes
        Instance of a clipping and noise estiamtion mode to use. Should return a BANEResult and ahve a .perform method. 
    max_loop : int, optional
        Maximum number of adjustable boxes to use. Will increase in steps of 100 pixels. If 0 adative mode is disabled, by default 5

    Returns
    -------
    Tuple[float,float]
        Backgrounf and noise estimation
    """
    
    
    original_box = box_size
    loop = 0
    while True:
        test_box_size = tuple([b+loop*100 for b in original_box])
        r_min, r_max, c_min, c_max = box(
            row, column, data_shape=data.shape, box_size=test_box_size
        )
        new = data[r_min:r_max, c_min:c_max]
        new = new.flatten()
        
        result = mode.perform(data=new)
    
        if loop >= max_loop:
            break
        if all(np.isfinite(result))  and result.valid_pixels > 0.9 * len(new):
            break
        
        loop += 1
        print(f"increasing, {loop=} {row=} {column=}")
        
    
    rms = result.rms
    bkg = result.bkg 
    
    return float(bkg), float(rms)
        

def sigma_filter(filename, region, step_size, box_size, shape, domask,
                 cube_index, mode, adaptive_loop=0):
    """
    Calculate the background and rms for a sub region of an image. The results
    are written to shared memory - irms and ibkg.

    Parameters
    ----------
    filename : string
        Fits file to open

    region : list
        Region within the fits file that is to be processed. (row_min,
        row_max).

    step_size : (int, int)
        The filtering step size

    box_size : (int, int)
        The size of the box over which the filter is applied (each step).

    shape : tuple
        The shape of the fits image

    domask : bool
        If true then copy the data mask to the output.

    cube_index : int
        The index into the 3rd dimension (if present)

    mode  : ClippingModes
        Which of the clipping modes to use. 
        
    adaptive_loop  : int
        The maximum number of resizes allowed should a box be resized. If 0 this is turned off.
        Default = 0

    Returns
    -------
    None
    """

    ymin, ymax = region
    logging.debug('rows {0}-{1} starting at {2}'.format(ymin,
                  ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    # cut out the region of interest plus 1/2 the box size
    # and clip to the image size
    data_row_min = max(0, ymin - box_size[0]//2)
    data_row_max = min(shape[0], ymax + box_size[0]//2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]

    # For some reason we can't memmap a file with BSCALE not 1.0
    # so we ignore it now and scale it later
    with fits.open(filename, memmap=True, do_not_scale_image_data=True) as a:
        if NAXIS == 2:
            data = a[0].section[data_row_min:data_row_max, 0:shape[1]]
        elif NAXIS == 3:
            data = np.squeeze(
                a[0].section[cube_index,
                             data_row_min:data_row_max, 0:shape[1]]
            )
        elif NAXIS == 4:
            data = np.squeeze(
                a[0].section[0, cube_index,
                             data_row_min:data_row_max, 0:shape[1]]
            )
        else:
            logging.error("Too many NAXIS for me {0}".format(NAXIS))
            logging.error("fix your file to be more sane")
            raise Exception("Too many NAXIS")

    # Manually scale the data if BSCALE is not 1.0
    header = fits.getheader(filename)
    if 'BSCALE' in header:
        data *= header['BSCALE']

    # force float64 for consistency
    data = data.astype(np.float64)

    # row_len = shape[1]

    logging.debug('data size is {0}'.format(data.shape))
    logging.debug('data format is {0}'.format(data.dtype))

    
    # set up a grid of rows/cols at which we will compute the bkg/rms
    rows = list(range(ymin-data_row_min, ymax-data_row_min, step_size[0]))
    rows.append(ymax-data_row_min)
    cols = list(range(0, shape[1], step_size[1]))
    cols.append(shape[1])

    # store the computed bkg/rms in this smaller array
    vals = np.zeros(shape=(len(rows), len(cols)))
    
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            
            bkg, rms = adaptive_box_estimate(
                data=data, row=row, column=col, box_size=box_size, mode=mode, max_loop=adaptive_loop
            )
            
            if np.isfinite(bkg):
                vals[i, j] = bkg

    # indices of all the pixels within our region
    gr, gc = np.mgrid[ymin-data_row_min:ymax-data_row_min, 0:shape[1]]

    # Find the shared memory and create a numpy array interface
    ibkg_shm = SharedMemory(name=f'ibkg_{memory_id}', create=False)
    ibkg = np.ndarray(shape, dtype=np.float64, buffer=ibkg_shm.buf)
    irms_shm = SharedMemory(name=f'irms_{memory_id}', create=False)
    irms = np.ndarray(shape, dtype=np.float64, buffer=irms_shm.buf)

    logging.debug("Interpolating bkg to sharemem")
    ifunc = RegularGridInterpolator((rows, cols), vals)
    interp_bkg = np.array(ifunc((gr, gc)), dtype=np.float64)
    ibkg[ymin:ymax, :] = interp_bkg
    del ifunc, interp_bkg
    logging.debug(" ... done writing bkg")

    # wait for all to complete
    i = barrier.wait()
    if i == 0:
        barrier.reset()

    logging.debug("background subtraction")
    data[0 + ymin - data_row_min: data.shape[0] -
         (data_row_max - ymax), :] -= ibkg[ymin:ymax, :]
    logging.debug(".. done ")

    # reset/recycle the vals array
    vals[:] = 0

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            rms = np.nan

            bkg, rms = adaptive_box_estimate(
                data=data, row=row, column=col, box_size=box_size, mode=mode, max_loop=adaptive_loop
            )
            
            if np.isfinite(rms):
                vals[i, j] = rms

    logging.debug("Interpolating rms to sharemem")
    ifunc = RegularGridInterpolator((rows, cols), vals)
    interp_rms = np.array(ifunc((gr, gc)), dtype=np.float64)
    irms[ymin:ymax, :] = interp_rms
    del ifunc, interp_rms
    logging.debug(" .. done writing rms")

    if domask:
        # wait for all to complete
        i = barrier.wait()
        if i == 0:
            barrier.reset()

        logging.debug("applying mask")
        mask = ~np.isfinite(
            data[0 + ymin - data_row_min: data.shape[0] -
                 (data_row_max - ymax), :])
        ibkg[ymin:ymax, :][mask] = np.nan
        irms[ymin:ymax, :][mask] = np.nan
        logging.debug("... done applying mask")
    logging.debug('rows {0}-{1} finished at {2}'.format(ymin,
                  ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    return


def filter_mc_sharemem(filename, step_size, box_size, cores, shape,
                       nslice=None, domask=True,
                       cube_index=0, mode: str='sigmaclip', mode_kwargs: Optional[Dict[str, Any]]=None,
                       adaptive_box: bool=False):
    """
    Calculate the background and noise images corresponding to the input file.
    The calculation is done via a box-car approach and uses multiple cores and
    shared memory.

    Parameters
    ----------
    filename : str
        Filename to be filtered.

    step_size : (int, int)
        Step size for the filter.

    box_size : (int, int)
        Box size for the filter.

    cores : int
        Number of cores to use. If None then use all available.

    nslice : int
        The image will be divided into this many horizontal stripes for
        processing. Default = None = equal to cores

    shape : (int, int)
        The shape of the image in the given file.

    domask : bool
        True(Default) = copy data mask to output.

    cube_index : int
        For 3d data use this index into the third dimension. Default = 0

    mode  : str
        Which background and rms estimation mode to use. 
        Default = 'sigmaclip', which is the original BANE mode. 
        
    adaptive_box  : bool
         Resize the box-car should a position be deemed to have unreliable
         statistics. Default = False. 


    Returns
    -------
    bkg, rms : numpy.ndarray
        The interpolated background and noise images.
    """
    mode_kwargs = mode_kwargs if mode_kwargs else {}

    if cores is None:
        cores = multiprocessing.cpu_count()
    if (nslice is None) or (cores == 1):
        nslice = cores

    img_y, img_x = shape

    logging.info("using {0} cores".format(cores))
    logging.info("using {0} stripes".format(nslice))

    if nslice > 1:
        # box widths should be multiples of the step_size, and not zero
        width_y = int(max(img_y/nslice/step_size[1], 1) * step_size[1])

        # locations of the box edges
        ymins = list(range(0, img_y, width_y))
        ymaxs = list(range(width_y, img_y, width_y))
        ymaxs.append(img_y)
    else:
        ymins = [0]
        ymaxs = [img_y]

    logging.debug("ymins {0}".format(ymins))
    logging.debug("ymaxs {0}".format(ymaxs))
    
    
    if mode.lower() not in AVAILABLE_MODES:
        raise ValueError(f"Mode not recognised. Received {mode=}, available modes {AVAILABLE_MODES}")

    logging.info(f"Using bane {mode=}")
    mode: ClippingModes = BANE_MODE_MAPPINGS[mode.lower()](**mode_kwargs)

    adaptive_loop = 5 if adaptive_box else 0
    logging.info(f"Adaptive box resize loops: {adaptive_loop}")

    args = []
    for region in zip(ymins, ymaxs):
        args.append((filename, region, step_size, box_size,
                    shape, domask, cube_index, mode, adaptive_loop))

    exit = False
    try:
        global memory_id
        memory_id = str(uuid.uuid4())[:15]
        nbytes = np.prod(shape) * np.float64(1).nbytes
        ibkg = SharedMemory(name=f'ibkg_{memory_id}', create=True, size=nbytes)
        irms = SharedMemory(name=f'irms_{memory_id}', create=True, size=nbytes)

        # start a new process for each task, hopefully to reduce residual
        # memory use
        method = 'spawn'
        if sys.platform.startswith('linux'):
            method = 'fork'
        ctx = multiprocessing.get_context(method)
        barrier = ctx.Barrier(parties=len(ymaxs))
        pool = ctx.Pool(processes=cores, maxtasksperchild=1,
                        initializer=init, initargs=(barrier, memory_id))
        try:
            # chunksize=1 ensures that we only send a single task to each
            # process
            pool.map_async(_sf2, args, chunksize=1).get(timeout=10000000)
        except KeyboardInterrupt:
            logging.error("Caught keyboard interrupt")
            pool.close()
            exit = True
        except Exception as e:
            logging.error(e)
        else:
            pool.close()
            pool.join()
            bkg = np.ndarray(shape, buffer=ibkg.buf,
                             dtype=np.float64).astype(np.float32)
            rms = np.ndarray(shape, buffer=irms.buf,
                           dtype=np.float64).astype(np.float32)
    
    except Exception as e:
        print(e)
    finally:
        ibkg.close()
        ibkg.unlink()
        irms.close()
        irms.unlink()
        if exit:
            sys.exit(1)
    return bkg, rms


def filter_image(im_name, out_base, step_size=None, box_size=None,
                 twopass=False,  # Deprecated
                 cores=None, mask=True, compressed=False, nslice=None,
                 cube_index=None, mode='sigmaclip', adaptive_box: bool=False):
    """
    Create a background and noise image from an input image. Resulting images
    are written to `outbase_bkg.fits` and `outbase_rms.fits`

    Parameters
    ----------
    im_name : str
        Image to filter.

    out_base : str or None
        The output filename base. Will be modified to make _bkg and _rms files.
        If None, then no files are written.

    step_size : (int,int)
        Tuple of the x,y step size in pixels

    box_size : (int,int)
        The size of the box in pixels

    twopass : bool
        Perform a second pass calculation to ensure that the noise is not
        contaminated by the background. Default = False. DEPRECATED

    cores : int
        Number of CPU corse to use. Default = all available

    nslice : int
        The image will be divided into this many horizontal stripes for
        processing. Default = None = equal to cores

    mask : bool
        Mask the output array to contain np.nna wherever the input array is nan
        or not finite. Default = true

    compressed : bool
        Return a compressed version of the background/noise images. Default =
        False

    cube_index : int
        If the input data is 3d, then use this index for the 3rd dimension.
        Default = None, use the first index.

    mode  : str
        Which background and rms estimation mode to use. 
        Default = 'sigmaclip', which is the original BANE mode. 
        
    adaptive_box  : bool
         Resize the box-car should a position be deemed to have unreliable
         statistics. Default = False. 

    Returns
    -------
    bkg, rms : `numpy.ndarray`
        The computed background and rms maps (not compressed)
    """
    # Use the first slice of the 3rd dimension if not specified
    if cube_index is None:
        cube_index = 0

    header = fits.getheader(im_name)
    shape = (header['NAXIS2'], header['NAXIS1'])
    naxis = header['NAXIS']
    if naxis > 2:
        naxis3 = header['NAXIS3']
        if cube_index >= naxis3:
            logging.error(
                "3rd dimension has len {0} but index {1} was passed".format(
                    naxis3, cube_index)
            )
            return None

    if step_size is None:
        step_size = get_step_size(header)

    if box_size is None:
        # default to 6x the step size so we have ~ 30beams
        box_size = (step_size[0]*6, step_size[1]*6)

    if compressed:
        if not step_size[0] == step_size[1]:
            step_size = (min(step_size), min(step_size))
            logging.info(
                "Changing grid to be {0} so we can compress the output".format(
                    step_size)
            )

    logging.info("using grid_size {0}, box_size {1}".format(
                 step_size, box_size))
    logging.info("on data shape {0}".format(shape))
    bkg, rms = filter_mc_sharemem(im_name,
                                  step_size=step_size, box_size=box_size,
                                  cores=cores, shape=shape, nslice=nslice,
                                  domask=mask, cube_index=cube_index, mode=mode,
                                  adaptive_box=adaptive_box)
    logging.info("done")

    if out_base is not None:
        # add a comment to the fits header
        header['HISTORY'] = 'BANE {0}-({1})'.format(__version__, __date__)

        bkg_out = '_'.join([os.path.expanduser(out_base), 'bkg.fits'])
        rms_out = '_'.join([os.path.expanduser(out_base), 'rms.fits'])

        # Test for BSCALE and scale back if needed before we write to a file
        bscale = 1.0
        if 'BSCALE' in header:
            bscale = header['BSCALE']

        # compress
        if compressed:
            hdu = fits.PrimaryHDU(bkg/bscale)
            hdu.header = copy.deepcopy(header)
            hdulist = fits.HDUList([hdu])
            compress(hdulist, step_size[0], bkg_out)
            hdulist[0].header = copy.deepcopy(header)
            hdulist[0].data = rms/bscale
            compress(hdulist, step_size[0], rms_out)
        else:
            write_fits(bkg/bscale, header, bkg_out)
            write_fits(rms/bscale, header, rms_out)

    return bkg, rms


###
# Helper functions
###
def get_step_size(header):
    """
    Determine the grid spacing for BANE operation.

    This is set to being 4x the synthesized beam width.
    If the beam is not circular then the "width" is sqrt(a*b)

    For the standard 4 pix/beam, the step size will be 16 pixels.

    Parameters
    ----------
    header

    Returns
    -------
    step_size : (int, int)
        The grid spacing for BANE operation
    """
    if 'BMAJ' in header and 'BMIN' in header:
        beam_size = np.sqrt(abs(header['BMAJ']*header['BMIN']))
        if 'CDELT1' in header:
            pix_scale = np.sqrt(abs(header['CDELT1']*header['CDELT2']))
        elif 'CD1_1' in header:
            pix_scale = np.sqrt(abs(header['CD1_1']*header['CD2_2']))
            if 'CD1_2' in header and 'CD2_1' in header:
                if header['CD1_2'] != 0 or header['CD2_1'] != 0:
                    logging.warning(
                        "CD1_2 and/or CD2_1 are non-zero and " +
                        "I don't know what to do with them")
                    logging.warning("Ingoring them")
        else:
            logging.warning(
                "Cannot determine pixel scale, assuming 4 pixels per beam")
            pix_scale = beam_size/4.
        # default to 4x the synthesized beam width
        step_size = int(np.ceil(4*beam_size/pix_scale))
    else:
        logging.info("BMAJ and/or BMIN not in fits header.")
        logging.info("Assuming 4 pix/beam, so we have step_size = 16 pixels")
        step_size = 16
    step_size = (step_size, step_size)
    return step_size


def write_fits(data, header, file_name):
    """
    Combine data and a fits header to write a fits file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be written.

    header : astropy.io.fits.hduheader
        The header for the fits file.

    file_name : string
        The file to write

    Returns
    -------
    None
    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_name, overwrite=True)
    logging.info("Wrote {0}".format(file_name))
    return
