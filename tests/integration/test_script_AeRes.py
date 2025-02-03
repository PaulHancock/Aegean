#! /usr/bin/env python

from AegeanTools.CLI import AeRes
import os
import sys

image_SIN = "tests/test_files/1904-66_SIN.fits"
catfile = "tests/test_files/1904_comp.fits"
tempfile = "dlme"


def test_help():
    sys.argv = ["--help"]
    AeRes.main()


def test_nocat():
    try:
        sys.argv = [""]  # no arguments
        AeRes.main()

        sys.argv = ["", "-c", catfile]
        AeRes.main()

        sys.argv = ["", "-c", catfile, "-f", image_SIN]
        AeRes.main()

        sys.argv = [
            "",
            "-c",
            catfile,
            "-f",
            image_SIN,
            "-r",
            tempfile,
            "-m",
            tempfile + "_model",
        ]
        AeRes.main()
    finally:
        if os.path.exists(tempfile):
            os.remove(tempfile)
        if os.path.exists(tempfile + "_model"):
            os.remove(tempfile + "_model")


def test_configfile():
    cfile = "AeRes.ini"
    try:
        with open(cfile, "w") as config:
            config.write("debug = False")
        sys.argv = ["", "--config", cfile]
        AeRes.main()
    finally:
        if os.path.exists(cfile):
            os.remove(cfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
