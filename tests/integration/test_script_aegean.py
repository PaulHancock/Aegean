#! /usr/bin/env python

from treasure_island.CLI import aegean
import os

image_SIN = 'tests/test_files/1904-66_SIN.fits'
image_AIT = 'tests/test_files/1904-66_AIT.fits'
tempfile = 'dlme'


def test_help():
    aegean.main()


def test_cite():
    aegean.main(['--cite'])


def test_table_formats():
    aegean.main(['--tformats'])


def test_versions():
    aegean.main(['--versions'])


def test_invalid_image():
    if not aegean.main(['none']):
        raise AssertionError('tried to run on invalid image')


def test_check_projection():
    aegean.main([image_AIT, '--nopositive'])


def test_turn_on_find():
    aegean.main([image_SIN, '--save'])


def test_debug():
    aegean.main(['--debug', image_SIN, '--save'])


def test_beam():
    aegean.main([image_SIN, '--save', '--beam', '1', '1', '0'])


def test_autoload():
    aegean.main([image_SIN, '--autoload', '--save'])


def test_aux_images():
    aegean.main([image_SIN, '--background', 'none', '--save'])
    aegean.main([image_SIN, '--noise', 'none', '--save'])
    aegean.main([image_SIN, '--psf', 'none', '--save'])
    aegean.main([image_SIN, '--catpsf', 'none', '--save'])
    aegean.main([image_SIN, '--region', 'none', '--save'])


def test_find():
    aegean.main([image_SIN, '--table', 'test'])
    aegean.main([image_SIN, '--out', 'stdout'])
    aegean.main([image_SIN, '--out', tempfile, '--blank'])
    os.remove(tempfile)


def test_priorized():
    aegean.main([image_SIN, '--table', tempfile + '.fits'])
    aegean.main([image_SIN, '--priorized', '1', '--ratio', '-1'])
    aegean.main([image_SIN, '--priorized', '3',
                '--ratio', '0.8'])
    aegean.main([image_SIN, '--priorized', '2',
                '--input', 'none'])
    aegean.main([image_SIN, '--priorized', '1', '--input',
                tempfile+'_comp.fits', '--out', 'stdout', '--island'])


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
