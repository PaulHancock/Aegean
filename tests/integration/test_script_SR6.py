#! /usr/bin/env python

from AegeanTools.CLI import SR6
import os

image_SIN = 'tests/test_files/1904-66_SIN.fits'
maskfile = 'tests/test_files/mask.fits'
tempfile = 'dlme'


def test_main():
    SR6.main()


def test_cite():
    SR6.main(['--cite'])


def test_no_infile():
    if os.path.exists(tempfile):
        os.remove(tempfile)
    SR6.main([tempfile])


def test_compress():
    SR6.main([image_SIN, '-o', tempfile])
    os.remove(tempfile)


def test_expand():
    SR6.main([image_SIN, '-o', tempfile+".fits"])
    SR6.main([tempfile+".fits", '-x', '-o', tempfile])
    SR6.main([tempfile+".fits",
              '-x',
              '-m', maskfile,
              '-o', tempfile])

    os.remove(tempfile)
    SR6.main([tempfile+".fits",
              '-x',
              '-m', tempfile,
              '-o', tempfile])
    os.remove(tempfile+".fits")


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
