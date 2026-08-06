#! /usr/bin/env python
"""
Test BANE.py
"""
import os
import signal
import multiprocessing as mp

import numpy as np
from astropy.io import fits
import AegeanTools
from AegeanTools import BANE

__author__ = "Paul Hancock"
import os
import signal
import multiprocessing as mp
from AegeanTools import BANE


def _flaky_worker_entrypoint(queue):
    """
    Run BANE.filter_image with exactly one worker rigged to fail partway
    through its computation, and report back what happened. Runs in its
    own process group so a stuck run (and all of its grandchild pool
    workers) can be cleaned up reliably from the test if needed.
    """
    os.setsid()

    counter = mp.Value("i", 0)
    orig_sigmaclip = BANE.sigmaclip

    def flaky_sigmaclip(arr, lo, hi, reps=10):
        with counter.get_lock():
            counter.value += 1
            should_fail = counter.value == 1
        if should_fail:
            raise RuntimeError("TEST: simulated failure in one BANE worker")
        return orig_sigmaclip(arr, lo, hi, reps)

    BANE.sigmaclip = flaky_sigmaclip
    try:
        BANE.filter_image(
            "tests/test_files/1904-66_SIN_3d.fits",
            out_base=None,
            cores=3,
            cube_index=None,
        )
        queue.put(("no_exception", None))
    except Exception as e:
        queue.put(("exception", f"{type(e).__name__}: {e}"))


def test_barrier_abort_on_worker_failure():
    TIMEOUT = 15
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_flaky_worker_entrypoint, args=(queue,))
    proc.start()
    proc.join(timeout=TIMEOUT)

    if proc.is_alive():
        # kill the whole process group, not just proc itself, since
        # filter_image spawns its own grandchild pool workers that
        # proc.terminate() alone would leave orphaned and still stuck.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.join()
        raise AssertionError(
            f"filter_image did not return within {TIMEOUT}s after a "
            "simulated worker failure -- sibling workers are likely stuck "
            "at a Barrier that was never aborted."
        )

    if queue.empty():
        raise AssertionError("worker process exited without reporting a result")

    kind, detail = queue.get()
    print("RESULT:", kind, detail)
    if kind != "exception":
        raise AssertionError(f"expected filter_image to raise, but got: {kind}")


def test_sigmaclip():
    """Test the sigmaclipping"""
    
    data = np.ones(100)
    bkg, rms = BANE.sigmaclip(data, 3, 4, reps=4)
    if not np.isclose(bkg, 1.0):
        raise AssertionError("BKG is not 1.0, it is {0}".format(bkg))
    if not np.isclose(rms, 0.0):
        raise AssertionError("RMS is not 0.0, it is {0}".format(rms))

    data[13] = np.nan
    bkg, rms = BANE.sigmaclip(data, 3, 4, reps=4)
    if not np.isclose(bkg, 1.0):
        raise AssertionError("BKG is not 1.0, it is {0}".format(bkg))
    if not np.isclose(rms, 0.0):
        raise AssertionError("RMS is not 0.0, it is {0}".format(rms))
    
    # test empty list
    if not np.isnan(BANE.sigmaclip(np.array([]), 0, 3)[0]):
        raise AssertionError()

    np.random.seed(42)
    data =np.random.normal(3.5, 0.2, size=100)
    data[0] = 11  # outlier to be clipped
    data[-1] = -3  # outlier to be clipped
    bkg, rms = BANE.sigmaclip(data, 3, 3, reps=4)
    if not np.isclose(bkg, 3.5, atol=0.05):
        raise AssertionError("BKG is not 3.5, it is {0}".format(bkg))
    if not np.isclose(rms, 0.2, atol=0.05):
        raise AssertionError("RMS is not 0.2, it is {0}".format(rms))
    

def test_filter_image():
    """Test filter image"""
    # data = np.random.random((30, 30), dtype=np.float32)
    fname = "tests/test_files/1904-66_SIN.fits"
    outbase = "dlme"
    rms = outbase + "_rms.fits"
    bkg = outbase + "_bkg.fits"
    # hdu = fits.getheader(fname)
    # shape = hdu[0]['NAXIS1'], hdu[0]['NAXIS2']
    BANE.filter_image(
        fname, step_size=(10, 10), box_size=(100, 100), cores=2, out_base=outbase
    )
    if not os.path.exists(rms):
        raise AssertionError()

    os.remove(rms)
    if not os.path.exists(bkg):
        raise AssertionError()

    os.remove(bkg)
    BANE.filter_image(fname, cores=2, out_base=outbase, compressed=True)
    if not os.path.exists(rms):
        raise AssertionError()

    os.remove(rms)
    if not os.path.exists(bkg):
        raise AssertionError()

    os.remove(bkg)


def test_ND_images():
    """Test that ND images are treated correctly"""
    fbase = "tests/test_files/small_{0}D.fits"
    outbase = "dlme"
    rms = outbase + "_rms.fits"
    bkg = outbase + "_bkg.fits"
    # this should work just fine, but trigger different NAXIS checks
    for fname in [fbase.format(n) for n in [3, 4]]:
        BANE.filter_image(fname, out_base=outbase)
        os.remove(rms)
        os.remove(bkg)

    fname = fbase.format(5)
    try:
        BANE.filter_image(fname, out_base=outbase)
    except Exception as e:
        pass
    else:
        raise AssertionError("BANE failed on 5d image in unexpected way")


def test_slice():
    """
    Test the BANE will give correct results when run with the slice option
    """
    fname = "tests/test_files/1904-66_SIN_3d.fits"
    # don't crash and die
    try:
        for index in [0, 1, 2]:
            BANE.filter_image(fname, out_base=None, nslice=1, cube_index=0)
    except Exception as e:
        raise AssertionError("Error on cube_index {0}:\n{1}".format(index, e))

    # die niecely when using an invalid cube_index
    try:
        BANE.filter_image(fname, out_base=None, nslice=1, cube_index=3)
    except Exception as e:
        raise AssertionError("BANE didn't die safely with cube_index=3")


def test_quantitative():
    """Test that the images are equal to a pre-calculated version"""
    fbase = "tests/test_files/1904-66_SIN"
    outbase = "dlme"
    BANE.filter_image(fbase + ".fits", out_base=outbase, cores=2, nslice=2)

    rms = outbase + "_rms.fits"
    bkg = outbase + "_bkg.fits"
    ref_rms = fbase + "_rms.fits"
    ref_bkg = fbase + "_bkg.fits"

    r1 = fits.getdata(rms)
    r2 = fits.getdata(ref_rms)
    b1 = fits.getdata(bkg)
    b2 = fits.getdata(ref_bkg)
    os.remove(rms)
    os.remove(bkg)
    if not np.allclose(r1, r2, atol=0.01, equal_nan=True):
        raise AssertionError("rms is wrong {0}".format(np.nanmax(np.abs(r1 - r2))))

    if not np.allclose(b1, b2, atol=0.01, equal_nan=True):
        raise AssertionError("bkg is wrong {0}".format(np.nanmax(np.abs(b1 - b2))))

    return


def test_BSCALE():
    """Test that BSCALE present and not 1.0 is handled properly"""
    fbase = "tests/test_files/1904-66_SIN"
    outbase = "dlme"
    hdu = fits.open(fbase + ".fits")
    hdu[0].header["BSCALE"] = 1.0
    hdu.writeto("dlme.fits")
    try:
        BANE.filter_image(outbase + ".fits", out_base=outbase, cores=1, nslice=1)
    except ValueError:
        raise AssertionError("BSCALE=1.0 causes crash")
    finally:
        os.remove("dlme.fits")

    hdu[0].header["BSCALE"] = 2.0
    hdu.writeto("dlme.fits")
    try:
        BANE.filter_image(outbase + ".fits", out_base=outbase, cores=1, nslice=1)
    except ValueError:
        raise AssertionError("BSCALE=2.0 causes crash")
    finally:
        os.remove("dlme.fits")
    return


def test_cube_as_cube():
    """
    Ensure that running BANE on a cube delivers a cube output
    """
    # the _3d image is the same as the base image, but I have added a third axis
    # the third axis values are the same as the base but I have added a slice number as an offset
    fname = "tests/test_files/1904-66_SIN_3d.fits"
    outbase = "dlme"
    rms_file = outbase + "_rms.fits"
    bkg_file = outbase + "_bkg.fits"
    ref_rms_file = "tests/test_files/1904-66_SIN_rms.fits"
    ref_bkg_file = "tests/test_files/1904-66_SIN_bkg.fits"

    AegeanTools.logging.logger.setLevel("DEBUG")
    BANE.filter_image(fname, out_base=outbase, cores=3, cube_index=None)
    rms = fits.getdata(rms_file)
    ref_rms = fits.getdata(ref_rms_file)
    bkg = fits.getdata(bkg_file)
    ref_bkg = fits.getdata(ref_bkg_file)

    # os.remove(rms_file)
    # os.remove(bkg_file)
    for slice in [0, 1, 2]:
        if not np.allclose(rms[slice], ref_rms, atol=0.01, equal_nan=True):
            raise AssertionError(f"rms is wrong on slice {slice} max diff is {np.nanmax(np.abs(rms[slice] - ref_rms))}")

        if not np.allclose(bkg[slice]-slice, ref_bkg, atol=0.01, equal_nan=True):
            raise AssertionError(f"bkg is wrong on slice {slice} max diff is {np.nanmax(np.abs(bkg[slice]-slice - ref_bkg))}")
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
