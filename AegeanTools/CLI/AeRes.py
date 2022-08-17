#! /usr/bin/env python

import argparse
import logging

from AegeanTools.AeRes import make_residual

__author__ = 'Paul Hancock'
__version__ = 'v0.2.7'
__date__ = '2020-07-30'


# global constants


def main(argv=()):
    """
    Tool for making residual images with Aegean tables as input
    """

    parser = argparse.ArgumentParser(prog='AeRes', prefix_chars='-')
    group1 = parser.add_argument_group("I/O arguments")
    group1.add_argument("-c", "--catalog", dest='catalog', default=None,
                        help="Catalog in a format that Aegean understands."
                             "\nRA/DEC should be in degrees, a/b/pa should be "
                             "in arcsec/arcsec/degrees.")
    group1.add_argument("-f", "--fitsimage", dest='fitsfile', default=None,
                        help="Input fits file.")
    group1.add_argument("-r", "--residual", dest='rfile', default=None,
                        help="Output residual fits file.")
    group1.add_argument('-m', "--model", dest='mfile', default=None,
                        help="Output model file [optional].")

    group2 = parser.add_argument_group("Config options")
    group2.add_argument('--add', dest='add', default=False,
                        action='store_true',
                        help="Add components instead of subtracting them.")
    group2.add_argument('--mask', dest='mask', default=False,
                        action='store_true',
                        help="Instead of subtracting sources, just mask them")
    group2.add_argument('--sigma', dest='sigma', default=4, type=float,
                        help='If masking, pixels above this SNR are masked'
                             '(requires input catalogue to list rms)')
    group2.add_argument('--frac', dest='frac', default=0, type=float,
                        help='If masking, pixels above frac*peak_flux are'
                             ' masked for each source')

    group3 = parser.add_argument_group("Catalogue options")
    group3.add_argument('--racol', dest='ra_col', default='ra',
                        help="RA column name")
    group3.add_argument('--deccol', dest='dec_col', default='dec',
                        help="Dec column name")
    group3.add_argument('--peakcol', dest='peak_col', default='peak_flux',
                        help="Peak flux column name")
    group3.add_argument('--acol', dest='a_col', default='a',
                        help="Major axis column name")
    group3.add_argument('--bcol', dest='b_col', default='b',
                        help="Minor axis column name")
    group3.add_argument('--pacol', dest='pa_col', default='pa',
                        help="Position angle column name")

    group4 = parser.add_argument_group("Extra options")
    group4.add_argument('--debug', dest='debug', action='store_true',
                        default=False, help="Debug mode.")

    options = parser.parse_args(args=argv)

    logging_level = logging.DEBUG if options.debug else logging.INFO
    logging.basicConfig(level=logging_level,
                        format="%(process)d:%(levelname)s %(message)s")
    logging.info("This is AeRes {0}-({1})".format(__version__, __date__))

    if options.catalog is None:
        logging.error("input catalog is required")
        parser.print_help()
        return 1
    if options.fitsfile is None:
        logging.error("input fits file is required")
        parser.print_help()
        return 1
    if options.rfile is None:
        logging.error("output residual filename is required")
        parser.print_help()
        return 1
    # convert default value of 0 to be None.
    if options.frac <= 0:
        options.frac = None

    logging.info("Using {0} and {1} to make {2}".format(
        options.fitsfile, options.catalog, options.rfile))
    if options.mfile is not None:
        logging.info(" and writing model to {0}".format(options.mfile))

    colmap = {'ra_col': options.ra_col,
              'dec_col': options.dec_col,
              'peak_col': options.peak_col,
              'a_col': options.a_col,
              'b_col': options.b_col,
              'pa_col': options.pa_col}

    make_residual(options.fitsfile, options.catalog, options.rfile,
                  mfile=options.mfile, add=options.add, mask=options.mask,
                  frac=options.frac, sigma=options.sigma,
                  colmap=colmap)
    return 0
