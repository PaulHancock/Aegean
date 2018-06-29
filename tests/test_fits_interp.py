#! python
__author__ = 'Paul Hancock'
__date__ = ''
from AegeanTools import fits_interp
from astropy.io import fits
import os


def test_load_file_or_hdu():
    """Test that we can 'open' either a file or HDU"""
    fname = 'tests/test_files/1904-66_AIT.fits'
    hdulist = fits.open(fname)
    if not (fits_interp.load_file_or_hdu(hdulist) is hdulist): raise AssertionError()


def test_compress():
    """Test the compression functionality"""
    # return None when the factor is not a positive integer
    if not (fits_interp.compress(None, factor=-1) is None): raise AssertionError()
    if not (fits_interp.compress(None, factor=0.3) is None): raise AssertionError()
    if not (fits_interp.compress(None, factor=0) is None): raise AssertionError()
    # test with factor = 10 for speed
    fname = 'tests/test_files/1904-66_AIT.fits'
    hdulist = fits.open(fname)
    cd1 = hdulist[0].header['CDELT1']
    cd2 = hdulist[0].header['CDELT2']
    # compress using CDELT1 and CDELT2
    if not (isinstance(fits_interp.compress(hdulist, factor=10), fits.HDUList)): raise AssertionError()
    hdulist[0].header['CD1_1'] = cd1
    del hdulist[0].header['CDELT1']
    hdulist[0].header['CD2_2'] = cd2
    del hdulist[0].header['CDELT2']
    # compress using CD1_1 and CDELT2_2 instead
    if not (isinstance(fits_interp.compress(hdulist, factor=10), fits.HDUList)): raise AssertionError()
    # now strip CD2_2 and we should get error
    del hdulist[0].header['CD2_2']
    if not (fits_interp.compress(hdulist, factor=10) is None): raise AssertionError()
    # same for CD1_1
    del hdulist[0].header['CD1_1']
    if not (fits_interp.compress(hdulist, factor=10) is None): raise AssertionError()


def test_expand():
    """Test the expand function"""
    fname = 'tests/test_files/1904-66_AIT.fits'
    hdulist = fits.open(fname)
    compressed = fits_interp.compress(hdulist, factor=10)
    # the uncompressed hdu list is missing header keys so test that this gives the expected result
    if not (fits_interp.expand(compressed) is hdulist): raise AssertionError()

    # now mix up the CDELT and CD keys
    compressed = fits_interp.compress(fname, factor=10)  # reload because we pass references
    compressed[0].header['CD1_1'] = compressed[0].header['CDELT1']
    del compressed[0].header['CDELT1']
    compressed[0].header['CD2_2'] = compressed[0].header['CDELT2']
    del compressed[0].header['CDELT2']
    if not (isinstance(fits_interp.expand(compressed), fits.HDUList)): raise AssertionError()

    # now strip CD2_2 and we should return None
    compressed = fits_interp.compress(fname, factor=10)
    del compressed[0].header['CDELT2']
    if not (fits_interp.expand(compressed) is None): raise AssertionError()
    # same for CD1_1
    del compressed[0].header['CDELT1']
    if not (fits_interp.expand(compressed) is None): raise AssertionError()


def test_fits_interp_compress_then_expand():
    """Test that we can interp/compress files"""
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
            globals()[f]()