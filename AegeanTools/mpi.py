#!/usr/bin/env python

MPI = None

try:
    import mpi4py
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI_AVAIL = True
    else:
        MPI_AVAIL = False
except ImportError:
    MPI_AVAIL = False