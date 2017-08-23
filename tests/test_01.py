#! python
from __future__ import print_function

from AegeanTools import BANE
import numpy as np

__author__ = 'Paul Hancock'
__date__ = ''


def test_sigmaclip():
    data = np.random.random(100)
    print("TESTING BANE.sigmaclip")
    BANE.sigmaclip(data, 3, 4, reps=4)
    print("Pass")

if __name__ == "__main__":
    test_sigmaclip()