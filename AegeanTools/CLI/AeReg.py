#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np
from astropy.io import fits

from AegeanTools import wcs_helpers
from AegeanTools.catalogs import load_table, save_catalog, table_to_source_list
from AegeanTools.cluster import check_attributes_for_regroup, regroup_dbscan, resize
from AegeanTools.logging import logger, logging

__author__ = ["PaulHancock"]
__date__ = "2024-01-24"
__version__ = "0.9"


def main():
    """
    A regrouping tool to accompany the Aegean source finding program.
    """

    parser = argparse.ArgumentParser(prog="regroup", prefix_chars="-")
    group1 = parser.add_argument_group("Required")
    group1.add_argument(
        "--input", dest="input", type=str, required=True, help="The input catalogue."
    )
    group1.add_argument(
        "--table",
        dest="tables",
        type=str,
        required=True,
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

    options = parser.parse_args()

    invocation_string = " ".join(sys.argv)

    # configure logging
    logging_level = logging.DEBUG if options.debug else logging.INFO
    logger.setLevel(logging_level)
    logger.info(f"This is regroup {__version__}-({__date__})")
    logger.debug(f"Run as:\n{invocation_string}")

    # check that a valid intput filename was entered
    filename = options.input
    if not os.path.exists(filename):
        logger.error(f"{filename} not found")
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
        logger.debug(f"{len(sources)} sources resized")
    elif options.ratio is not None:
        sources = resize(sources, ratio=options.ratio)
        logger.debug(f"{len(sources)} sources resized")
    else:
        logger.debug("Not rescaling")

    if options.regroup:
        if not check_attributes_for_regroup(sources):
            logger.error("Cannot use catalog")
            return 1
        logger.debug(f"Regrouping with eps={options.eps}[arcmin]")
        eps = np.sin(np.radians(options.eps / 60))
        groups = regroup_dbscan(sources, eps=eps)
        sources = [source for group in groups for source in group]
        logger.debug(f"{len(sources)} sources regrouped")
    else:
        logger.debug("Not regrouping")

    if options.tables:
        meta = {
            "PROGRAM": "regroup",
            "PROGVER": f"{__version__}-({__date__})",
            "CATFILE": filename,
            "RUN-AS": invocation_string,
        }
        for t in options.tables.split(","):
            logger.debug(f"writing {t}")
            save_catalog(t, sources, meta=meta)
    return 0
