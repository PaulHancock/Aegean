#! /usr/bin/env python
import sys

from AegeanTools.CLI import BANE

image_SIN = "tests/test_files/1904-66_SIN.fits"
image_AIT = "tests/test_files/1904-66_AIT.fits"
tempfile = "dlme"


def test_help():
    sys.argv = [""]
    BANE.main()


def test_cite():
    sys.argv = ["", "--cite"]
    BANE.main()


def test_invalid_file():
    sys.argv = ["", tempfile]
    BANE.main()


def test_noclobber():
    sys.argv = ["", "--noclobber", image_SIN]
    BANE.main()


def test_run_BANE():
    sys.argv = ["", image_SIN]
    BANE.main()


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
