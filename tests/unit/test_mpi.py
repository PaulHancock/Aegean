#!/usr/bin/env python

def test_import_mpi():
    try:
        from AegeanTools.mpi import MPI_AVAIL, MPI
    except Exception:
        raise AssertionError("Error importing AegeanTools.mpi")
    

    try:
        import mpi4py
        if MPI is None:
            raise AssertionError("MPI should not be None when mpi4py is available")
    except ImportError:
        pass