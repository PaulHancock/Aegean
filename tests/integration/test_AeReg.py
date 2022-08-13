#! /usr/bin/env python

from AegeanTools.CLI import AeReg
import os

image_SIN = 'tests/test_files/1904-66_SIN.fits'
catfile = 'tests/test_files/1904_comp.fits'
mimfile = 'tests/test_files/1904-66_SIN.mim'
regfile = 'tests/test_files/ds9.reg'
maskfile = 'tests/test_files/mask.fits'
tempfile = 'dlme'


def no_test_help():
    AeReg.main(['--help'])


def test_input():
    AeReg.main(['--input', tempfile, '--table ', tempfile])


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
