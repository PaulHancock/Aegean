#! python
from __future__ import print_function

from AegeanTools import BANE
from AegeanTools import fits_interp
import numpy as np
import os

__author__ = 'Paul Hancock'
__date__ = '23/08/2017'

def test_sigmaclip():
    # normal usage case
    data = np.random.random(100)
    data[13] = np.nan
    assert len(BANE.sigmaclip(data, 3, 4, reps=4)) > 0

    # test list where all elements get clipped
    assert len(BANE.sigmaclip([-10, 10], 1, 2, reps=2)) == 0

    # test empty list
    assert len(BANE.sigmaclip([], 0, 3)) == 0


def test_optimum_sections():
    # typical case
    assert BANE.optimum_sections(8, (64, 64)) == (2, 4)
    # redundant case
    assert BANE.optimum_sections(1, (134, 1200)) == (1, 1)


def test_mask_data():
    data = np.ones((10, 10), dtype=np.float32)
    mask = data.copy()
    mask[3:5, 0:2] = np.nan
    BANE.mask_img(data, mask)
    # check that the nan regions overlap
    assert np.all(np.isnan(data) == np.isnan(mask))


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")