#! python
__author__ = 'Paul Hancock'
__date__ = ''
from AegeanTools import fits_interp
from astropy.io import fits
import os


def test_load_file_or_hdu():
    fname = 'tests/test_files/1904-66_AIT.fits'
    hdulist = fits.open(fname)
    assert fits_interp.load_file_or_hdu(hdulist) is hdulist


def test_compress():
    # return None when the factor is not a positive integer
    assert fits_interp.compress(None, factor=-1) is None
    assert fits_interp.compress(None, factor=0.3) is None
    assert fits_interp.compress(None, factor=0) is None
    # test with factor = 10 for speed
    fname = 'tests/test_files/1904-66_AIT.fits'
    hdulist = fits.open(fname)
    cd1 = hdulist[0].header['CDELT1']
    cd2 = hdulist[0].header['CDELT2']
    # compress using CDELT1 and CDELT2
    assert isinstance(fits_interp.compress(hdulist, factor=10), fits.HDUList)
    hdulist[0].header['CD1_1'] = cd1
    del hdulist[0].header['CDELT1']
    hdulist[0].header['CD2_2'] = cd2
    del hdulist[0].header['CDELT2']
    # compress using CD1_1 and CDELT2_2 instead
    assert isinstance(fits_interp.compress(hdulist, factor=10), fits.HDUList)
    # now strip CD2_2 and we should get error
    del hdulist[0].header['CD2_2']
    assert fits_interp.compress(hdulist, factor=10) is None
    # same for CD1_1
    del hdulist[0].header['CD1_1']
    assert fits_interp.compress(hdulist, factor=10) is None


def test_expand():
    fname = 'tests/test_files/1904-66_AIT.fits'
    hdulist = fits.open(fname)
    compressed = fits_interp.compress(hdulist, factor=10)
    # the uncompressed hdu list is missing header keys so test that this gives the expected result
    assert fits_interp.expand(hdulist) is hdulist
    # now mix up the CDELT and CD keys
    cd1 = compressed[0].header['CDELT1']
    cd2 = compressed[0].header['CDELT2']
    compressed[0].header['CD1_1'] = cd1
    del compressed[0].header['CDELT1']
    compressed[0].header['CD2_2'] = cd2
    del compressed[0].header['CDELT2']
    assert isinstance(fits_interp.expand(compressed), fits.HDUList)
    # now strip CD2_2 and we should get error
    del compressed[0].header['CD2_2']
    assert fits_interp.compress(compressed, factor=10) is None
    # same for CD1_1
    del compressed[0].header['CD1_1']
    assert fits_interp.compress(compressed, factor=10) is None




def test_fits_interp_compress_then_expand():
    # test compression and expansion
    fits_interp.compress("tests/test_files/1904-66_AIT.fits", factor=7, outfile='tests/temp/test.fits')
    fits_interp.expand('tests/temp/test.fits', outfile='tests/temp/test2.fits', method='linear')
    # cleanup
    os.remove('tests/temp/test.fits')
    os.remove('tests/temp/test2.fits')


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")