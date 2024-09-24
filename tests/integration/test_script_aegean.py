#! /usr/bin/env python

from AegeanTools.CLI import aegean
import os
import sys

image_SIN = "tests/test_files/1904-66_SIN.fits"
image_AIT = "tests/test_files/1904-66_AIT.fits"
tempfile = "dlme"


def test_help():
    sys.argv = [""]
    aegean.main()


def test_cite():
    sys.argv = ["", "--cite"]
    aegean.main()


def test_table_formats():
    sys.argv = ["", "--tformats"]
    aegean.main()


def test_versions():
    sys.argv = ["", "--versions"]
    aegean.main()


def test_invalid_image():
    sys.argv = ["", "none"]
    if not aegean.main():
        raise AssertionError("tried to run on invalid image")


def test_check_projection():
    sys.argv = ["", image_AIT, "--nopositive"]
    aegean.main()


def test_turn_on_find():
    sys.argv = ["", image_SIN, "--save"]
    aegean.main()


def test_debug():
    sys.argv = ["", "--debug", image_SIN, "--save"]
    aegean.main()


def test_beam():
    sys.argv = ["", image_SIN, "--save", "--beam", "1", "1", "0"]
    aegean.main()


def test_autoload():
    sys.argv = ["", image_SIN, "--autoload", "--save"]
    aegean.main()


def test_aux_images():
    for flag in ["--backgroun", "--noise", "--psf", "--catpsf", "--region"]:
        sys.argv = ["", image_SIN, flag, "none", "--save"]
        aegean.main()


def test_find():
    # sys.argv = ["", image_SIN, "--table", "test"]
    # aegean.main()

    sys.argv = ["", image_SIN, "--out", "stdout"]
    aegean.main()

    sys.argv = ["", image_SIN, "--out", tempfile, "--blank"]
    aegean.main()
    os.remove(tempfile)


def test_priorized():
    sys.argv = ["", image_SIN, "--table", tempfile + ".fits"]
    aegean.main()

    sys.argv = ["", image_SIN, "--priorized", "1", "--ratio", "-1"]
    aegean.main()

    sys.argv = ["", image_SIN, "--priorized", "3", "--ratio", "0.8"]
    aegean.main()

    sys.argv = ["", image_SIN, "--priorized", "2", "--input", "none"]
    aegean.main()

    sys.argv = [
        "",
        image_SIN,
        "--priorized",
        "1",
        "--input",
        tempfile + "_comp.fits",
        "--out",
        "stdout",
        "--island",
    ]
    aegean.main()


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
