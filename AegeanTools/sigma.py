"""Tooling around the rms and background estimation"""

from re import L
from typing import Tuple, NamedTuple, Optional
import logging 

import numpy as np
from scipy.stats import norm
from astropy.stats import sigma_clip

class FittedSigmaClip(NamedTuple):
    """Arguments for the fitted_sigma_clip"""
    sigma: int = 3 
    """Threshhold before clipped"""

def fitted_mean(data: np.ndarray, axis: Optional[int] =None) -> float:
    if axis is not None:
        # This is to make astropy sigma clip happy
        raise NotImplementedError("Unexpected axis keyword. ")
    
    mean, _ = norm.fit(data)
    
    return mean


def fitted_std(data: np.ndarray, axis: Optional[int]=None) -> float:
    if axis is not None:
        # This is to make astropy sigma clip happy
        raise NotImplementedError("Unexpected axis keyword. ")
    
    _, std = norm.fit(data)
    
    return std

def fitted_sigma_clip(data: np.ndarray, sigma: int=3) -> Tuple[float,float]:
    
    data = data[np.isfinite(data)]
    
    clipped_plane = sigma_clip(
        data.flatten(), 
        sigma=3, 
        cenfunc=np.median, 
        stdfunc=fitted_std, 
        maxiters=None
    )
    bkg, rms = norm.fit(clipped_plane.compressed())

    return float(bkg), float(rms)

class FitBkgRmsEstimate(NamedTuple):
    """Options for the fitting approach method"""
    clip_rounds: int = 3
    """Number of clipping rounds to perform"""
    bin_perc: float = 0.25
    """Minimum fraction of the histogram bins, or something"""
    outlier_thres: float = 3.0
    """Threshold that a data point should be at to be considered an outlier"""

def mad(data, bkg=None):
    bkg = bkg if bkg else np.median(data)
    return np.median(np.abs(data - bkg))

def fit_bkg_rms_estimate(
    data: np.ndarray,
    clip_rounds: int = 2,
    bin_perc: float = 0.25,
    outlier_thres: float = 3.0,
) -> Tuple[float,float]:
    
    data = data[np.isfinite(data)]

    cen_func = np.median

    bkg = cen_func(data)

    for i in range(clip_rounds):
        data = data[np.abs(data - bkg) < outlier_thres * mad(data, bkg=bkg)]
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

    return float(bkg), noise



class SigmaClip(NamedTuple):
    """Container for the original sigma clipping method"""
    low: float = 3.0
    """Low sigma clip threshhold"""
    high: float = 3.0
    """High sigma clip threshhold"""

def sigmaclip(arr, lo, hi, reps=10):
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

    return mean, std