#! /usr/bin/env python

from treasure_island.CLI import AeRes
import os

image_SIN = 'tests/test_files/1904-66_SIN.fits'
catfile = 'tests/test_files/1904_comp.fits'
tempfile = 'dlme'


def no_test_help():
    AeRes.main(['--help'])


def test_nocat():
    AeRes.main()
    AeRes.main(['-c', catfile])
    AeRes.main(['-c', catfile,
                '-f', image_SIN])
    AeRes.main(['-c', catfile,
                '-f', image_SIN,
                '-r', tempfile,
                '-m', tempfile+'_model'])
    os.remove(tempfile)
    os.remove(tempfile+'_model')


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
