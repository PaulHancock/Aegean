#! /usr/bin/env python
from __future__ import annotations

from treasure_island.CLI import BANE

image_SIN = 'tests/test_files/1904-66_SIN.fits'
image_AIT = 'tests/test_files/1904-66_AIT.fits'
tempfile = 'dlme'


def test_help():
    BANE.main()


def test_cite():
    BANE.main(['--cite'])


def test_invalid_file():
    BANE.main([tempfile])


def test_noclobber():
    BANE.main(['--noclobber', image_SIN])


def test_run_BANE():
    BANE.main([image_SIN])


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
