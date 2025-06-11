#! /usr/bin/env python

from treasure_island.CLI import AeReg
import os

image_SIN = 'tests/test_files/1904-66_SIN.fits'
catfile = 'tests/test_files/1904_comp.fits'
shortcat = 'tests/test_files/1904_incomplete_cat.csv'
mimfile = 'tests/test_files/1904-66_SIN.mim'
regfile = 'tests/test_files/ds9.reg'
maskfile = 'tests/test_files/mask.fits'
tempfile = 'dlme'


def no_test_help():
    AeReg.main(['--help'])


def test_input():
    # file not found
    AeReg.main(['--input', tempfile, '--table', tempfile])


def test_noregroup():
    # should run
    AeReg.main(['--input', catfile, '--table',
               tempfile+'.csv', '--noregroup'])
    os.remove(tempfile+'_comp.csv')


def test_regroup():
    AeReg.main(['--input', catfile, '--table',
               tempfile+'.csv', '--debug', '--eps', '1'])
    os.remove(tempfile+'_comp.csv')

    AeReg.main(['--input', catfile, '--table',
               tempfile+'.csv', '--ratio', '1.2'])
    os.remove(tempfile+'_comp.csv')

    AeReg.main(['--input', catfile, '--table',
               tempfile+'.csv', '--ratio', '1.2',
               '--psfheader', image_SIN])
    os.remove(tempfile+'_comp.csv')


def test_broken_catalogue():
    # this still works due to the way that the sources
    # are loaded into a table
    AeReg.main(['--input', shortcat, '--table',
               tempfile, '--debug'])
    os.remove(tempfile+'_comp')


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
