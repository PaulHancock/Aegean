#! /usr/bin/env python

"""
A program to provide a command line interface to the AegeanTools.fits_interp module

@author: Paul Hancock

Created:
3rd Feb 2015
"""

import logging
import argparse
import os
import sys

from AegeanTools.fits_interp import reduce, expand

version = '1.0'

# command line version of this program runs from here.
if __name__ == "__main__":

    epilog = ''
    parser = argparse.ArgumentParser(epilog=epilog, prefix_chars='-')

    # tools for shrinking files
    group1 = parser.add_argument_group("Shrinking and expanding files")
    group1.add_argument('infile', type=str,
                        help="input filename")
    group1.add_argument('-o', dest='outfile', action='store',
                        default=None, type=str, metavar='outputfile',
                        help='output filename')
    group1.add_argument('-f', dest='factor', action='store',
                        default=None, type=int, metavar='factor',
                        help='reduction factor')
    group1.add_argument('-x', dest='expand', action='store_true',
                        default=False,
                        help='Operation is expand instead of compress.')
    # TODO: move these to be in a different group. (The same as help).
    group1.add_argument('--debug', dest='debug', action='store_true',
                        default=False,
                        help='Debug output')
    group1.add_argument('--version', action='version', version='%(prog)s '+version)

    results = parser.parse_args()

    logging_level = logging.DEBUG if results.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is SR6 {0}".format(version))


    if not os.path.exists(results.infile):
        logging.error("{0} does not exist".format(results.infile))
        sys.exit()

    if results.expand:
        expand(results.infile, results.outfile)
    else:
        reduce(results.infile, results.factor, results.outfile)
