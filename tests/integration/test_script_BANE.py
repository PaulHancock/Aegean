#! /usr/bin/env python
import os
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


def test_configfile():
    """Test that we can use a config file"""
    cfile = "BANE.ini"
    with open(cfile, "w") as config:
        config.write("cite = True")
    sys.argv = ["", "--config", cfile]
    try:
        BANE.main()
    finally:
        if os.path.exists(cfile):
            os.remove(cfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
