#! /usr/bin/env python

from AegeanTools.CLI import aegean
import os
import sys

image_SIN = "tests/test_files/1904-66_SIN.fits"
image_AIT = "tests/test_files/1904-66_AIT.fits"
input_cat = "tests/test_files/1904_comp.fits"
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
    for flag in ["--background", "--noise", "--psf", "--region"]:
        sys.argv = ["", image_SIN, flag, "none", "--save"]
        aegean.main()


def test_find():
    # sys.argv = ["", image_SIN, "--table", "test"]
    # aegean.main()

    sys.argv = ["", image_SIN, "--out", "stdout"]
    print(f"Running aegean {' ' .join(sys.argv)}")
    aegean.main()

    sys.argv = ["", image_SIN, "--out", tempfile, "--blank"]
    print(f"Running aegean {' ' .join(sys.argv)}")
    aegean.main()
    os.remove(tempfile)


def test_priorized_fail_on_negative_ratio():
    sys.argv = ["", image_SIN, "--priorized", "1", "--ratio", "-1"]
    print(f"Running aegean {' ' .join(sys.argv)}")
    try:
        aegean.main()
    except SystemExit as e:
        if e.code != 1:
            raise AssertionError("aegean did not exit with code 1 on negative ratio")


def test_priorized_warning_on_fractional_ratio():
    sys.argv = ["", image_SIN, "--priorized", "1", "--ratio", "0.5"]
    print(f"Running aegean {' ' .join(sys.argv)}")
    try:
        aegean.main()
    except SystemExit as e:
        if e.code != 0:
            raise AssertionError("aegean did not exit with code 0 on fractional ratio")


def test_priorized_fail_on_nonexistent_input():
    sys.argv = ["", image_SIN, "--priorized", "1", "--input", "none"]
    print(f"Running aegean {' ' .join(sys.argv)}")
    try:
        aegean.main()
    except SystemExit as e:
        if e.code != 1:
            raise AssertionError(
                "aegean did not exit with code 1 on nonexistent input"
            )


def test_priorized_normal():
    sys.argv = [
        "",
        image_SIN,
        "--priorized",
        "1",
        "--input",
        input_cat,
        "--out",
        "stdout",
        "--island",
    ]
    print(f"Running aegean {' ' .join(sys.argv)}")
    aegean.main()


def test_priorized3D():
    sys.argv = [
        "",
        "tests/test_files/synthetic_cube.fits",
        "--noise",
        "tests/test_files/synthetic_cube_rms.fits",
        "--background",
        "tests/test_files/synthetic_cube_bkg.fits",
        "--table",
        "output.csv",
        "--priorized",
        "1",
        "--input",
        "tests/test_files/synthetic_cat_no_alpha_comp.fits",
    ]
    aegean.main()

    if not os.path.exists("output_comp.csv"):
        raise AssertionError("output file not created")
    else:
        os.remove("output_comp.csv")


def test_find3D():
    sys.argv = [
        "",
        "tests/test_files/synthetic_cube.fits",
        "--noise",
        "tests/test_files/synthetic_cube_rms.fits",
        "--background",
        "tests/test_files/synthetic_cube_bkg.fits",
        "--table",
        "output.csv",
    ]
    aegean.main()

    if not os.path.exists("output_comp.csv"):
        raise AssertionError("output file not created")
    else:
        os.remove("output_comp.csv")


def test_configfile():
    cfile = "aegean.ini"
    try:
        with open(cfile, "w") as config:
            config.write("debug = False")
        sys.argv = ["", "--config", cfile]
        aegean.main()
    finally:
        if os.path.exists(cfile):
            os.remove(cfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
