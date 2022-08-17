#! /usr/bin/env python
import argparse
import logging
import os

import numpy as np
from AegeanTools import __citation__
from AegeanTools.BANE import get_step_size
from AegeanTools.fits_tools import compress, expand
from astropy.io import fits

__author__ = "Paul Hancock"
__version__ = 'v1.2'
__date__ = '2022-07-15'


# command line version of this program runs from here.
def main(argv=()):
    """
    A program to provide a command line interface to the
    AegeanTools.fits_tools module
    """

    epilog = ''
    parser = argparse.ArgumentParser(epilog=epilog, prefix_chars='-')

    # tools for shrinking files
    group1 = parser.add_argument_group("Shrinking and expanding files")
    group1.add_argument('infile', type=str, nargs='?', default=None,
                        help="input filename")
    group1.add_argument('-o', dest='outfile', action='store',
                        default=None, type=str, metavar='OutputFile',
                        help='output filename')
    group1.add_argument('-f', dest='factor', action='store',
                        default=None, type=int, metavar='factor',
                        help='reduction factor. Default is 4x psf.')
    group1.add_argument('-x', dest='expand', action='store_true',
                        default=False,
                        help='Operation is expand instead of compress.')
    group1.add_argument('-m', dest='maskfile', action='store',
                        default=None, type=str, metavar='MaskFile',
                        help="File to use for masking pixels.")
    group2 = parser.add_argument_group("Other options")
    group2.add_argument('--debug', dest='debug', action='store_true',
                        default=False,
                        help='Debug output')
    group2.add_argument('--version', action='version',
                        version='%(prog)s {0}-({1})'.format(__version__,
                                                            __date__))
    group2.add_argument('--cite', dest='cite', action="store_true",
                        default=False,
                        help='Show citation information.')

    results = parser.parse_args(args=argv)
    # print help if the user enters no options or filename
    if len(argv) == 0:
        parser.print_help()
        return 0

    if results.cite:
        print(__citation__)
        return 0

    logging_level = logging.DEBUG if results.debug else logging.INFO
    logging.basicConfig(level=logging_level,
                        format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is SR6 {0}-({1})".format(__version__, __date__))

    if not os.path.exists(results.infile):
        logging.error("{0} does not exist".format(results.infile))
        return 1

    if results.expand:
        if results.maskfile and os.path.exists(results.maskfile):
            maskdata = fits.open(results.maskfile)[0].data
            mask = np.where(np.isnan(maskdata))
            hdulist = expand(results.infile)
            hdulist[0].data[mask] = np.nan
            hdulist.writeto(results.outfile, overwrite=True)
            logging.info("Wrote masked file: {0}".format(results.outfile))
        elif results.maskfile is None:
            expand(results.infile, results.outfile)
        else:
            logging.error("Can't find {0}".format(results.maskfile))

    else:
        if results.factor is None:
            header = fits.getheader(results.infile)
            results.factor = get_step_size(header)[0]
            logging.info(
                "Using compression factor of {0}".format(results.factor))
        compress(results.infile, results.factor, results.outfile)
