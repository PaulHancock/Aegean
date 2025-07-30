#! /usr/bin/env python

from AegeanTools.CLI import AeRes
import os
import sys

image_2d = "tests/test_files/synthetic_image.fits"
image_3d = "tests/test_files/synthetic_cube.fits"
cat_with_alpha = "tests/test_files/synthetic_cat_with_alpha_comp.fits"
cat_no_alpha = "tests/test_files/synthetic_cat_no_alpha_comp.fits"
# image_SIN = "tests/test_files/1904-66_SIN.fits"
# catfile = "tests/test_files/1904_comp.fits"
tempfile = "dlme"


def test_help():
    sys.argv = ["--help"]
    AeRes.main()


def test_cat_with_alpha_2d_image():
    try:
        sys.argv = ["", "-c", cat_with_alpha, "-f", image_2d, "-r", tempfile]
        AeRes.main()
    finally:
        if os.path.exists(tempfile):
            os.remove(tempfile)
        else:
            raise Exception("Output file not created")


def test_cat_with_alpha_3d_image():
    try:
        sys.argv = ["", "-c", cat_with_alpha, "-f", image_3d, "-r", tempfile]
        AeRes.main()
    finally:
        if os.path.exists(tempfile):
            os.remove(tempfile)
        else:
            raise Exception("Output file not created")


def test_cat_no_alpha_2d_image():
    try:
        sys.argv = ["", "-c", cat_no_alpha, "-f", image_2d, "-r", tempfile]
        AeRes.main()
    finally:
        if os.path.exists(tempfile):
            os.remove(tempfile)
        else:
            raise Exception("Output file not created")


def test_cat_no_alpha_3d_image():
    try:
        sys.argv = ["", "-c", cat_no_alpha, "-f", image_3d, "-r", tempfile]
        AeRes.main()
    finally:
        if os.path.exists(tempfile):
            os.remove(tempfile)
        else:
            raise Exception("Output file not created")


def test_nocat():
    try:
        sys.argv = [""]  # no arguments
        AeRes.main()

        sys.argv = ["", "-c", cat_no_alpha]
        AeRes.main()

        sys.argv = ["", "-c", cat_no_alpha, "-f", image_2d]
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
