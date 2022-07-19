#!/usr/bin/env python
"""
Script to fix the BEAM info for images created by AIPS
Will read beam info from HISTORY and put it into the correct fits keywords
"""

__author__ = ["Guo Shaoguang", "Paul Hancock"]
__version__ = 'v1.0'
__date__ = '2016-09-29'
__institute__ = 'Shanghai Astronomical Observatory'

import sys
import argparse
from AegeanTools.fits_tools import load_file_or_hdu
from AegeanTools.wcs_helpers import fix_aips_header


def search_beam(hdulist):
    """
    Will search the beam info from the HISTORY
    :param hdulist:
    :return:
    """
    header = hdulist[0].header
    history = header['HISTORY']
    history_str = str(history)
    # AIPS   CLEAN BMAJ=  1.2500E-02 BMIN=  1.2500E-02 BPA=   0.00
    if 'BMAJ' in history_str:
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', dest='infile', default=None,
                        required=True,
                        help='The input fits file', metavar='<filename>')

    parser.add_argument('-o', '--outfile', dest='outfile', default='out.fits',
                        help='The output fits file', metavar='out.fits')
    results = parser.parse_args()

    if not results.infile:
        parser.print_help()
        sys.exit()

    print("Updating {0} -> {1}".format(results.infile, results.outfile))

    hdulist = load_file_or_hdu(results.infile)
    found = search_beam(hdulist)

    if found:
        fix_aips_header(hdulist[0].header)
        hdulist[0].header['HISTORY'] = "fix_beam.py by {0}".format(
            __institute__)
        print('Header has been updated')
        hdulist.writeto(results.outfile, clobber=True)
        print("Wrote {0}".format(results.outfile))
    else:
        print('Header has not been updated')
