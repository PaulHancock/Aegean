#! /usr/bin/env python

"""
Module for reading at writing catalogs
"""

__author__ = "Paul Hancock"

# input/output table formats
from astropy.table.table import Table
from astropy.io import ascii
from astropy.io import fits

try:
    from astropy.io.votable import from_table, parse_single_table
    from astropy.io.votable import writeto as writetoVO
    votables_supported = True
except ImportError:
    votables_supported = False

try:
    import h5py
    hdf5_supported = True
except ImportError:
    hdf5_supported = False

import sqlite3

