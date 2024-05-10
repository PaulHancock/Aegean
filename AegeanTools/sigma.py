"""Tooling around the rms and background estimation"""

from re import L
from typing import Tuple, NamedTuple, Optional
import logging 

import numpy as np
from scipy.stats import norm
from astropy.stats import sigma_clip

class BANEResult(NamedTuple):
    """Container for function results"""
    rms: float
    """RMS constrained"""
    bkg: float 
    """BKG constrained"""
    valid_pixels: int 
    """Number of pixels constrained against"""

class FittedSigmaClip(NamedTuple):
    """Arguments for the fitted_sigma_clip"""
    sigma: int = 3 
    """Threshhold before clipped"""
    
    def perform(self, data: np.ndarray) -> BANEResult:
        return fitted_sigma_clip(data=data, sigma=self.sigma)
        
def fitted_mean(data: np.ndarray, axis: Optional[int] =None) -> float:
    """Internal function that returns the mean by fitting to pixel distribution data"""
    if axis is not None:
        # This is to make astropy sigma clip happy
        raise NotImplementedError("Unexpected axis keyword. ")
    
    mean, _ = norm.fit(data)
    
    return mean


def fitted_std(data: np.ndarray, axis: Optional[int]=None) -> float:
    """Internal function that retunrs the stf by fitting to the pixel distribution"""
    if axis is not None:
        # This is to make astropy sigma clip happy
        raise NotImplementedError("Unexpected axis keyword. ")
    
    _, std = norm.fit(data)
    
    return std

def fitted_sigma_clip(data: np.ndarray, sigma: int=3) -> BANEResult:
    """Estimate the back ground and noise level by fitting to the pixel distribution. 
    Sigma clipping is performed using the fitted statistics. 

    Parameters
    ----------
    data : np.ndarray
        Data that will be considered
    sigma : int, optional
        Threshold for a point to be flagged, by default 3

    Returns
    -------
    BANEResult
        RMS and bkg ground statistics
    """
    data = data[np.isfinite(data)]
    
    clipped_plane = sigma_clip(
        data.flatten(), 
        sigma=sigma, 
        cenfunc=fitted_mean, 
        stdfunc=fitted_std, 
        maxiters=None
    )
    bkg, rms = norm.fit(clipped_plane.compressed())

    result = BANEResult(rms=float(rms), bkg=float(bkg), valid_pixels=len(clipped_plane.compressed()))

    return result

class FitBkgRmsEstimate(NamedTuple):
    """Options for the fitting approach method"""
    clip_rounds: int = 3
    """Number of clipping rounds to perform"""
    bin_perc: float = 0.25
    """Minimum fraction of the histogram bins, or something"""
    outlier_thres: float = 3.0
    """Threshold that a data point should be at to be considered an outlier"""

    def perform(self, data: np.ndarray) -> BANEResult:
        return fit_bkg_rms_estimate(data=data, clip_rounds=self.clip_rounds, bin_perc=self.bin_perc, outlier_thres=self.outlier_thres)

def mad(data, bkg=None):
    """Compute the median asbolute deviation. optionally provide a 
    precomuted background measure
    """
    bkg = bkg if bkg else np.median(data)
    return np.median(np.abs(data - bkg))

def fit_bkg_rms_estimate(
    data: np.ndarray,
    clip_rounds: int = 2,
    bin_perc: float = 0.25,
    outlier_thres: float = 3.0,
) -> BANEResult:
    """An over the top attempt at robustly characterising the
    back ground and RMS. Data will first be flagged via the MAD,
    then bin, then those bin counts are used to fit for a gaussian. 
    
    Only bins that are above 25 percent of the maximum bin count 
    are used in the fitting process. 

    Parameters
    ----------
    data : np.ndarray
        Data to be estimated
    clip_rounds : int, optional
        Number of clipping rounds, by default 2
    bin_perc : float, optional
        Minimum set of counts, relative to the max count bin, that any bin should have. Less than this and they are discarded, by default 0.25
    outlier_thres : float, optional
        Clipping threshold to use, by default 3.0

    Returns
    -------
    BANEResult
        Backgroun and noise estimation
    """
    data = data[np.isfinite(data)]

    cen_func = np.median

    bkg = cen_func(data)

    for i in range(clip_rounds):
        data = data[np.abs(data - bkg) < outlier_thres * 1.4826 * mad(data, bkg=bkg)]
        bkg = cen_func(data)

    # Attempts to ensure a sane number of bins to fit against
    mask_counts = 0
    loop = 1
    while True:
        counts, binedges = np.histogram(data, bins=50 * loop)

        mask = counts >= bin_perc * np.max(counts)
        mask_counts = np.sum(mask)
        loop += 1

        if not (mask_counts < 5 and loop < 5): 
            break

    binc = (binedges[:-1] + binedges[1:]) / 2
    p = np.polyfit(binc[mask], np.log10(counts[mask] / np.max(counts)), 2)
    a, b, c = p

    x1 = (-b + np.sqrt(b ** 2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    x2 = (-b - np.sqrt(b ** 2 - 4.0 * a * (c - np.log10(0.5)))) / (2.0 * a)
    fwhm = np.abs(x1 - x2)
    noise = fwhm / 2.355

    result = BANEResult(rms=float(noise), bkg=float(bkg), valid_pixels=len(data))

    return result



class SigmaClip(NamedTuple):
    """Container for the original sigma clipping method"""
    low: float = 3.0
    """Low sigma clip threshhold"""
    high: float = 3.0
    """High sigma clip threshhold"""

    def perform(self, data: np.ndarray) -> BANEResult:
        return sigmaclip(arr=data, lo=self.low, hi=self.high)

def sigmaclip(arr, lo, hi, reps=10) -> BANEResult:      
    """
    Perform sigma clipping on an array, ignoring non finite values.

    During each iteration return an array whose elements c obey:
    mean -std*lo < c < mean + std*hi

    where mean/std are the mean std of the input array.

    Parameters
    ----------
    arr : iterable
        An iterable array of numeric types.
    lo : float
        The negative clipping level.
    hi : float
        The positive clipping level.
    reps : int
        The number of iterations to perform. Default = 3.

    Returns
    -------
    mean : float
        The mean of the array, possibly nan
    std : float
        The std of the array, possibly nan

    Notes
    -----
    Scipy v0.16 now contains a comparable method that will ignore nan/inf
    values.
    """
    clipped = np.array(arr)[np.isfinite(arr)]

    if len(clipped) < 1:
        return np.nan, np.nan

    std = np.std(clipped)
    mean = np.mean(clipped)
    prev_valid = len(clipped)
    for count in range(int(reps)):
        mask = (clipped > mean-std*lo) & (clipped < mean+std*hi)
        clipped = clipped[mask]

        curr_valid = len(clipped)
        if curr_valid < 1:
            break
        # No change in statistics if no change is noted
        if prev_valid == curr_valid:
            break
        std = np.std(clipped)
        mean = np.mean(clipped)
        prev_valid = curr_valid
    else:
        logging.debug(
            "No stopping criteria was reached after {0} cycles".format(count))

    result = BANEResult(rms=float(std), bkg=float(mean), valid_pixels=len(clipped))

    return result