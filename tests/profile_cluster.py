#! /usr/bin/env python
"""
Profile functions within the cluster module of AegeanTools
"""

from AegeanTools.cluster import regroup, regroup_vectorized
from AegeanTools import catalogs

import numpy as np
import cProfile
import pstats
import io
from pstats import SortKey


def get_cat():
    # create/load a catalogue for regrouping
    table = catalogs.load_table('test_GLEAMX_comp.fits')[:1000]
    srccat = catalogs.table_to_source_list(table)
    return srccat


def profile_regroup():
    srccat = get_cat()
    regroup(srccat, eps=np.sqrt(2))
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('profile'):
            print(f)
            globals()[f]()
