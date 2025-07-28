#! /usr/bin/env python

import configargparse
import os
import sys

import numpy as np
from astropy.io import fits

from AegeanTools import wcs_helpers
from AegeanTools.catalogs import load_table, save_catalog, table_to_source_list
from AegeanTools.cluster import check_attributes_for_regroup, regroup_dbscan, resize
from AegeanTools.logging import logger, logging

__author__ = ["PaulHancock"]
__version__ = "0.9.1"
__date__ = "2025-01-14"


def main():
    """
    A regrouping tool to accompany the Aegean source finding program.
    """

    parser = configargparse.ArgumentParser(prog="AeReg", prefix_chars="-")
    group1 = parser.add_argument_group("Required")
    group1.add_argument("--input", dest="input", type=str, help="The input catalogue.")
    group1.add_argument(
        "--table",
        dest="tables",
        type=str,
        help="Table outputs, format inferred from extension.",
    )

    group2 = parser.add_argument_group("Clustering options")
    group2.add_argument(
        "--eps",
        dest="eps",
        default=4,
        type=float,
        help="The grouping parameter epsilon (~arcmin)",
    )
    group2.add_argument(
        "--noregroup",
        dest="regroup",
        default=True,
        action="store_false",
        help="Do not perform regrouping (default False)",
    )

    group3 = parser.add_argument_group("Scaling options")
    group3.add_argument(
        "--ratio",
        dest="ratio",
        default=None,
        type=float,
        help="The ratio of synthesized beam sizes "
        + "(image psf / input catalog psf).",
    )
    group3.add_argument(
        "--psfheader",
        dest="psfheader",
        default=None,
        type=str,
        help="A file from which the *target* psf is read.",
    )

    group4 = parser.add_argument_group("Other options")
    group4.add_argument(
        "--debug", dest="debug", action="store_true", default=False, help="Debug mode."
    )
    group4.add_argument("--config", is_config_file=True, help="Path to the config file")

    options = parser.parse_args()

    if not options.input or not options.tables:
        parser.print_help()
        return 0

    invocation_string = " ".join(sys.argv)

    # configure logging
    logging_level = logging.DEBUG if options.debug else logging.INFO
    logger.setLevel(logging_level)
    logger.info("This is AeReg {0}-({1})".format(__version__, __date__))
    logger.debug("Run as:\n{0}".format(invocation_string))

    # check that a valid intput filename was entered
    filename = options.input
    if not os.path.exists(filename):
        logger.error("{0} not found".format(filename))
        return 1

    input_table = load_table(options.input)
    input_sources = np.array(table_to_source_list(input_table))

    sources = input_sources

    # Rescale before regrouping since the shape of a source
    # is used as part of the regrouping
    if (options.ratio is not None) and (options.psfheader is not None):
        logger.info("Both --ratio and --psfheader specified")
        logger.info("Ignoring --ratio")
    if options.psfheader is not None:
        head = fits.getheader(options.psfheader)
        wcshelper = wcs_helpers.WCSHelper.from_header(head)
        sources = resize(sources, wcshelper=wcshelper)
        logger.debug("{0} sources resized".format(len(sources)))
    elif options.ratio is not None:
        sources = resize(sources, ratio=options.ratio)
        logger.debug("{0} sources resized".format(len(sources)))
    else:
        logger.debug("Not rescaling")

    if options.regroup:
        if not check_attributes_for_regroup(sources):
            logger.error("Cannot use catalog")
            return 1
        logger.debug("Regrouping with eps={0}[arcmin]".format(options.eps))
        eps = np.sin(np.radians(options.eps / 60))
        groups = regroup_dbscan(sources, eps=eps)
        sources = [source for group in groups for source in group]
        logger.debug("{0} sources regrouped".format(len(sources)))
    else:
        logger.debug("Not regrouping")

    if options.tables:
        meta = {
            "PROGRAM": "regroup",
            "PROGVER": "{0}-({1})".format(__version__, __date__),
            "CATFILE": filename,
            "RUN-AS": invocation_string,
        }
        for t in options.tables.split(","):
            logger.debug("writing {0}".format(t))
            save_catalog(t, sources, meta=meta)
    return 0
