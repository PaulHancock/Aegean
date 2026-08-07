#!/usr/bin/env python

MPI = None
MPI_AVAIL = False

try:
    import mpi4py
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI_AVAIL = True
except ImportError:
    pass