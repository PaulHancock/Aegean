#! /usr/bin/env python

from AegeanTools.CLI import AeReg
import os
import sys

image_SIN = "tests/test_files/1904-66_SIN.fits"
catfile = "tests/test_files/1904_comp.fits"
shortcat = "tests/test_files/1904_incomplete_cat.csv"
mimfile = "tests/test_files/1904-66_SIN.mim"
regfile = "tests/test_files/ds9.reg"
maskfile = "tests/test_files/mask.fits"
tempfile = "dlme"


def no_test_help():
    sys.argv = ["", "--help"]
    AeReg.main()


def test_input():
    # file not found
    sys.argv = ["", "--input", tempfile, "--table", tempfile]
    AeReg.main()


def test_noregroup():
    # should run
    try:
        sys.argv = ["", "--input", catfile, "--table", tempfile + ".csv", "--noregroup"]
        AeReg.main()
    finally:
        if os.path.exists(tempfile + "_comp.csv"):
            os.remove(tempfile + "_comp.csv")


def test_regroup():
    try:
        sys.argv = [
            "",
            "--input",
            catfile,
            "--table",
            tempfile + ".csv",
            "--debug",
            "--eps",
            "1",
        ]
        AeReg.main()

        sys.argv = [
            "",
            "--input",
            catfile,
            "--table",
            tempfile + ".csv",
            "--ratio",
            "1.2",
        ]
        AeReg.main()

        sys.argv = [
            "",
            "--input",
            catfile,
            "--table",
            tempfile + ".csv",
            "--ratio",
            "1.2",
            "--psfheader",
            image_SIN,
        ]
        AeReg.main()
    finally:
        if os.path.exists(tempfile + "_comp.csv"):
            os.remove(tempfile + "_comp.csv")


def test_broken_catalogue():
    # this still works due to the way that the sources
    # are loaded into a table
    try:
        sys.argv = ["", "--input", shortcat, "--table", tempfile, "--debug"]
        AeReg.main()
    finally:
        if os.path.exists(tempfile + "_comp"):
            os.remove(tempfile + "_comp")


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
