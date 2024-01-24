#! /usr/bin/env python

from AegeanTools.CLI import SR6
import os
import sys

image_SIN = "tests/test_files/1904-66_SIN.fits"
maskfile = "tests/test_files/mask.fits"
tempfile = "dlme"


def test_main():
    sys.argv = [""]
    SR6.main()


def test_cite():
    sys.argv = ["", "--cite"]
    SR6.main()


def test_no_infile():
    if os.path.exists(tempfile):
        os.remove(tempfile)

    sys.argv = ["", tempfile]
    SR6.main()


def test_compress():
    try:
        sys.argv = ["", image_SIN, "-o", tempfile]
        SR6.main()
    finally:
        os.remove(tempfile)


def test_expand():
    try:
        sys.argv = ["", image_SIN, "-o", tempfile + ".fits"]
        SR6.main()

        sys.argv = ["", tempfile + ".fits", "-x", "-o", tempfile]
        SR6.main()

        sys.argv = ["", tempfile + ".fits", "-x", "-m", maskfile, "-o", tempfile]
        SR6.main()
    finally:
        os.remove(tempfile)

    try:
        sys.argv = ["", tempfile + ".fits", "-x", "-m", tempfile, "-o", tempfile]
        SR6.main()
    finally:
        os.remove(tempfile + ".fits")


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
