#! /usr/bin/env python

from AegeanTools.CLI import MIMAS
import os

image_SIN = 'tests/test_files/1904-66_SIN.fits'
catfile = 'tests/test_files/1904_comp.fits'
mimfile = 'tests/test_files/1904-66_SIN.mim'
regfile = 'tests/test_files/ds9.reg'
maskfile = 'tests/test_files/mask.fits'
tempfile = 'dlme'


def test_help():
    MIMAS.main()


def test_citation():
    MIMAS.main(['--cite'])


def test_fitsmask():
    MIMAS.main(['--fitsmask', '', '', ''])


def test_mim2reg():
    MIMAS.main(['--mim2reg', mimfile, tempfile])
    os.remove(tempfile)


def test_reg2mim():
    MIMAS.main(['--reg2mim', regfile, tempfile])
    os.remove(tempfile)


def test_mim2fits():
    MIMAS.main(['--mim2fits', mimfile, tempfile])
    os.remove(tempfile)


def test_area():
    MIMAS.main(['--area', mimfile])


def test_intersect():
    MIMAS.main(['--intersect', mimfile])
    MIMAS.main(['--intersect', mimfile, '--intersect', mimfile])
    MIMAS.main(['--intersect', mimfile, '--intersect', mimfile, '-o', tempfile])
    os.remove(tempfile)


def test_mask_image():
    MIMAS.main(['--maskimage', mimfile, image_SIN, tempfile])
    os.remove(tempfile)


def test_mask_cat():
    MIMAS.main(['--maskcat', mimfile, catfile, tempfile+'.csv'])
    os.remove(tempfile+'.csv')


def test_mask2mim():
    MIMAS.main(['--mask2mim', maskfile, tempfile])
    os.remove(tempfile)


def test_make_region():
    MIMAS.main(['+c', '290', '15', '1', '-g', '-depth', '3', '-o', tempfile])
    os.remove(tempfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
