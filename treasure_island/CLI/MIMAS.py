#! /usr/bin/env python
from __future__ import annotations

import argparse

from treasure_island import MIMAS


def main(argv=()):
    """
    MIMAS - The Multi-resolution Image Mask for Aegean Software
    """
    epilog = (
        "Regions are added/subtracted in the following order, "
        "+r -r +c -c +p -p. This means that you might have to take "
        "multiple passes to construct overly complicated regions."
    )
    parser = argparse.ArgumentParser(epilog=epilog, prefix_chars="+-")

    group1 = parser.add_argument_group(
        "Creating/modifying regions", "Must specify -o, plus or more [+-][cr]"
    )
    # tools for creating .mim files
    group1.add_argument(
        "-o", dest="outfile", action="store", help="output filename", default=None
    )
    group1.add_argument(
        "-depth",
        dest="maxdepth",
        action="store",
        metavar="N",
        default=8,
        type=int,
        help="maximum nside=2**N to be used to represent this region. [Default=8]",
    )
    group1.add_argument(
        "+r",
        dest="add_region",
        action="append",
        default=[],
        type=str,
        metavar="filename",
        nargs="*",
        help="add a region specified by the given file (.mim format)",
    )
    group1.add_argument(
        "-r",
        dest="rem_region",
        action="append",
        default=[],
        type=str,
        metavar="filename",
        nargs="*",
        help="exclude a region specified by the given file (.mim format)",
    )
    # add/remove circles
    group1.add_argument(
        "+c",
        dest="include_circles",
        action="append",
        default=[],
        type=float,
        metavar=("ra", "dec", "radius"),
        nargs=3,
        help="add a circle to this region (decimal degrees)",
    )
    group1.add_argument(
        "-c",
        dest="exclude_circles",
        action="append",
        default=[],
        type=float,
        metavar=("ra", "dec", "radius"),
        nargs=3,
        help="exclude the given circles from a region",
    )

    # add/remove polygons
    group1.add_argument(
        "+p",
        dest="include_polygons",
        action="append",
        default=[],
        type=float,
        metavar=("ra", "dec"),
        nargs="*",
        help="add a polygon to this region ( decimal degrees)",
    )
    group1.add_argument(
        "-p",
        dest="exclude_polygons",
        action="append",
        default=[],
        type=float,
        metavar=("ra", "dec"),
        nargs="*",
        help="remove a polygon from this region (decimal degrees)",
    )
    group1.add_argument(
        "-g",
        dest="galactic",
        action="store_true",
        default=False,
        help="Interpret input coordinates are galactic instead of equatorial.",
    )

    group2 = parser.add_argument_group("Using already created regions")
    # tools that use .mim files
    group2.add_argument(
        "--mim2reg",
        dest="mim2reg",
        action="append",
        type=str,
        metavar=("region.mim", "region.reg"),
        nargs=2,
        help="convert region.mim into region.reg",
        default=[],
    )
    group2.add_argument(
        "--reg2mim",
        dest="reg2mim",
        action="append",
        type=str,
        metavar=("region.reg", "region.mim"),
        nargs=2,
        help="Convert a .reg file into a .mim file",
        default=[],
    )
    group2.add_argument(
        "--mim2fits",
        dest="mim2fits",
        action="append",
        type=str,
        metavar=("region.mim", "region_MOC.fits"),
        nargs=2,
        help="Convert a .mim file into a MOC.fits file",
        default=[],
    )
    group2.add_argument(
        "--mask2mim",
        dest="mask2mim",
        action="store",
        type=str,
        metavar=("mask.fits", "region.mim"),
        nargs=2,
        help="Convert a masked image into a region file",
        default=[],
    )
    group2.add_argument(
        "--intersect",
        "+i",
        dest="intersect",
        action="append",
        type=str,
        default=[],
        metavar=("region.mim",),
        help="Write out the intersection of the given regions.",
    )
    group2.add_argument(
        "--area",
        dest="area",
        action="store",
        default=None,
        help="Report the area of a given region",
        metavar=("region.mim",),
    )
    group3 = parser.add_argument_group("Masking files with regions")
    group3.add_argument(
        "--maskcat",
        dest="mask_cat",
        action="store",
        type=str,
        metavar=("region.mim", "INCAT", "OUTCAT"),
        nargs=3,
        default=[],
        help="use region.mim as a mask on INCAT, writing OUTCAT",
    )
    group3.add_argument(
        "--maskimage",
        dest="mask_image",
        action="store",
        type=str,
        metavar=("region.mim", "file.fits", "masked.fits"),
        nargs=3,
        default=[],
        help="use region.mim to mask the image file.fits and write masekd.fits",
    )
    group3.add_argument(
        "--fitsmask",
        dest="fits_mask",
        action="store",
        type=str,
        metavar=("mask.fits", "file.fits", "masked_file.fits"),
        nargs=3,
        default=[],
        help="Use a fits file as a mask for another fits file."
        " Values of blank/nan/zero are considered to be "
        "mask=True.",
    )
    group3.add_argument(
        "--negate",
        dest="negate",
        action="store_true",
        default=False,
        help="By default all masks will exclude data that are "
        "within the given region. Use --negate to exclude"
        " data that is outside of the region instead.",
    )
    group3.add_argument(
        "--colnames",
        dest="radec_colnames",
        action="store",
        type=str,
        metavar=("RA_name", "DEC_name"),
        nargs=2,
        default=("ra", "dec"),
        help="The name of the columns which contain the RA/DEC data. Default=(ra,dec).",
    )

    group4 = parser.add_argument_group("Extra options")
    # extras
    group4.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=1.0,
        help="Threshold value for input mask file.",
    )
    group4.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="debug mode [default=False]",
        default=False,
    )
    group4.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + MIMAS.__version__ + f"-({MIMAS.__date__})",
    )
    group4.add_argument(
        "--cite",
        dest="cite",
        action="store_true",
        default=False,
        help="Show citation information.",
    )

    results = parser.parse_args(args=argv)

    if results.cite:
        return 0

    # get the MIMAS logger
    logging = MIMAS.logging
    logging_level = logging.DEBUG if results.debug else logging.INFO
    logging.basicConfig(
        level=logging_level, format="%(process)d:%(levelname)s %(message)s"
    )
    logging.info(f"This is MIMAS {MIMAS.__version__}-({MIMAS.__date__})")

    if len(results.fits_mask) > 0:
        logging.info("The --fitsmask option is not yet implemented.")
        return 1

    if len(results.mim2reg) > 0:
        for i, o in results.mim2reg:
            MIMAS.mim2reg(i, o)
        return 0

    if len(results.reg2mim) > 0:
        for i, o in results.reg2mim:
            MIMAS.reg2mim(i, o, results.maxdepth)
        return 0

    if len(results.mim2fits) > 0:
        for i, o in results.mim2fits:
            MIMAS.mim2fits(i, o)
        return 0

    if results.area is not None:
        region = MIMAS.Region.load(results.area)
        return 0

    if len(results.intersect) > 0:
        if len(results.intersect) == 1:
            return 1
        if results.outfile is None:
            return 1
        region = MIMAS.intersect_regions(results.intersect)
        MIMAS.save_region(region, results.outfile)
        return 0

    if len(results.mask_image) > 0:
        m, i, o = results.mask_image
        MIMAS.mask_file(m, i, o, results.negate)
        return 0

    if len(results.mask_cat) > 0:
        m, i, o = results.mask_cat
        racol, deccol = results.radec_colnames
        MIMAS.mask_catalog(m, i, o, results.negate, racol, deccol)
        return 0

    if len(results.mask2mim) > 0:
        maskfile, mimfile = results.mask2mim
        MIMAS.mask2mim(
            maskfile=maskfile,
            mimfile=mimfile,
            threshold=results.threshold,
            maxdepth=results.maxdepth,
        )
        return 0

    if results.outfile is not None:
        region = MIMAS.combine_regions(results)
        MIMAS.save_region(region, results.outfile)
        return None
    return None
