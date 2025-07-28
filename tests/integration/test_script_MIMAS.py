#! /usr/bin/env python

from AegeanTools.CLI import MIMAS
import os
import sys

image_SIN = "tests/test_files/1904-66_SIN.fits"
catfile = "tests/test_files/1904_comp.fits"
mimfile = "tests/test_files/1904-66_SIN.mim"
regfile = "tests/test_files/ds9.reg"
maskfile = "tests/test_files/mask.fits"
tempfile = "dlme"


def test_help():
    sys.argv = [""]
    MIMAS.main()


def test_citation():
    sys.argv = ["", "--cite"]
    MIMAS.main()


def test_fitsmask():
    sys.argv = ["", "--fitsmask", "", "", ""]
    MIMAS.main()


def test_mim2reg():
    try:
        sys.argv = ["", "--mim2reg", mimfile, tempfile]
        MIMAS.main()
    finally:
        os.remove(tempfile)


def test_reg2mim():
    try:
        sys.argv = ["", "--reg2mim", regfile, tempfile]
        MIMAS.main()
    finally:
        os.remove(tempfile)


def test_mim2fits():
    try:
        sys.argv = ["", "--mim2fits", mimfile, tempfile]
        MIMAS.main()
    finally:
        os.remove(tempfile)


def test_area():
    sys.argv = ["", "--area", mimfile]
    MIMAS.main()


def test_intersect():
    try:
        sys.argv = ["", "--intersect", mimfile]
        MIMAS.main()

        sys.argv = ["", "--intersect", mimfile, "--intersect", mimfile]
        MIMAS.main()

        sys.argv = ["", "--intersect", mimfile, "--intersect", mimfile, "-o", tempfile]
        MIMAS.main()
    finally:
        os.remove(tempfile)


def test_mask_image():
    try:
        sys.argv = ["", "--maskimage", mimfile, image_SIN, tempfile]
        MIMAS.main()
    finally:
        os.remove(tempfile)


def test_mask_cat():
    try:
        sys.argv = ["", "--maskcat", mimfile, catfile, tempfile + ".csv"]
        MIMAS.main()
    finally:
        os.remove(tempfile + ".csv")


def test_mask2mim():
    try:
        sys.argv = ["", "--mask2mim", maskfile, tempfile]
        MIMAS.main()
    finally:
        os.remove(tempfile)


def test_make_region():
    try:
        sys.argv = ["", "+c", "290", "15", "1", "-g", "-depth", "3", "-o", tempfile]
        MIMAS.main()
    finally:
        os.remove(tempfile)


def test_configfile():
    cfile = "MIMAS.ini"
    try:
        with open(cfile, "w") as config:
            config.write("debug = False")
        sys.argv = ["", "--config", cfile]
        MIMAS.main()
    finally:
        if os.path.exists(cfile):
            os.remove(cfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
