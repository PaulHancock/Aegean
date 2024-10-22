#! /usr/bin/env python

import argparse

# import logging.config
import multiprocessing
import os
import sys

import astropy
from astropy.table import vstack
import lmfit
import numpy as np
import scipy

from AegeanTools import __citation__, __date__, __version__
from AegeanTools.catalogs import check_table_formats, save_catalog, show_formats, load_table, write_table
from AegeanTools.logging import logger, logging
from AegeanTools.source_finder import get_aux_files
from AegeanTools.wcs_helpers import Beam
from AegeanTools.mpi import MPI_AVAIL, MPI
from AegeanTools.exceptions import AegeanSuffixError

header = """#Aegean version {0}
# on dataset: {1}"""

def addSuffix(file, suffix):
    """
    A function to add a specified suffix before the extension.

    parameters
    ----------
    file: str
        The current name of the file
    
    suffix: str or int
        The desired suffix to be inserted before the extension
    """
    if isinstance(suffix, int):
        base, ext = os.path.splitext(file)
        base += f"_{suffix:02d}"
        fname = base + ext
    elif isinstance(suffix, str):
        if suffix[0] == "_":
            suffix = suffix[1:]
        base, ext = os.path.splitext(file)
        base += f"_{suffix}"
        fname = base + ext
    else:
        raise AegeanSuffixError(f"This suffix type is not support: {suffix}") 

    return fname

def check_projection(filename, options, log=logger):
    """
    Kindly let the user know that projections other than SIN need some careful
    thought.

    parameters
    ----------
    filename : str
        The input fits filename

    options : argparse options
        Options from the command line
    """
    header = astropy.io.fits.getheader(filename)
    if not ("SIN" in header["CTYPE1"]):
        if options.imgpsf is None:
            projection = header["CTYPE1"].split("-")[-1]
            logger.warning(
                "For projection {0} you should consider supplying a psf via"
                " --psf".format(projection)
            )
    return


def main():
    """
    The Aegean source finding program.
    """

    parser = argparse.ArgumentParser(prog="aegean", prefix_chars="-")
    parser.add_argument("image", nargs="?", default=None)
    group1 = parser.add_argument_group("Configuration Options")

    group1.add_argument(
        "--find",
        dest="find",
        action="store_true",
        default=False,
        help="Source finding mode. [default: true, unless "
        "--save or --measure are selected]",
    )
    group1.add_argument(
        "--hdu",
        dest="hdu_index",
        type=int,
        default=0,
        help="HDU index (0-based) for cubes with multiple "
        "images in extensions. [default: 0]",
    )

    group1.add_argument(
        "--beam",
        dest="beam",
        type=float,
        nargs=3,
        default=None,
        help='The beam parameters to be used is "--beam major'
        ' minor pa" all in degrees. '
        "[default: read from fits header].",
    )
    group1.add_argument(
        "--slice",
        dest="slice",
        type=int,
        default=0,
        help="If the input data is a cube, then this slice "
        "will determine the array index of the image "
        "which will be processed by aegean",
    )
    group1.add_argument(
        "--progress",
        default=False,
        action="store_true",
        help="Provide a progress bar as islands are being fit." " [default: False]",
    )

    # Input
    group2 = parser.add_argument_group("Input Options")
    group2.add_argument(
        "--forcerms",
        dest="rms",
        type=float,
        default=None,
        help="Assume a single image noise of rms." " [default: None]",
    )
    group2.add_argument(
        "--forcebkg",
        dest="bkg",
        type=float,
        default=None,
        help="Assume a single image background of bkg." " [default: None]",
    )
    group1.add_argument(
        "--cores",
        dest="cores",
        type=int,
        default=None,
        help="Number of CPU cores to use when calculating "
        "background and rms images [default: all cores]",
    )
    group2.add_argument(
        "--noise",
        dest="noiseimg",
        default=None,
        type=str,
        help="A .fits file that represents the image noise "
        "(rms), created from Aegean with --save "
        "or BANE. [default: none]",
    )
    group2.add_argument(
        "--background",
        dest="backgroundimg",
        default=None,
        type=str,
        help="A .fits file that represents the background "
        "level, created from Aegean with --save "
        "or BANE. [default: none]",
    )
    group2.add_argument(
        "--psf",
        dest="imgpsf",
        default=None,
        type=str,
        help="A .fits file that represents the local PSF. ",
    )
    group2.add_argument(
        "--autoload",
        dest="autoload",
        action="store_true",
        default=False,
        help="Automatically look for background, noise, "
        "region, and psf files using the input filename "
        "as a hint. [default: don't do this]",
    )
    group2.add_argument(
        "--3d",
        dest="threeD",
        action="store_true",
        default=False,
        help="Treat the input image as a 3D cube. "
    )

    # Output
    group3 = parser.add_argument_group("Output Options")
    group3.add_argument(
        "--out",
        dest="outfile",
        default=None,
        type=str,
        help="Destination of Aegean catalog output. " "[default: No output]",
    )
    group3.add_argument(
        "--table",
        dest="tables",
        default=None,
        type=str,
        help="Additional table outputs, format inferred from "
        "extension. [default: none]",
    )
    group3.add_argument(
        "--tformats",
        dest="table_formats",
        action="store_true",
        default=False,
        help="Show a list of table formats supported in this "
        "install, and their extensions",
    )
    group3.add_argument(
        "--blankout",
        dest="blank",
        action="store_true",
        default=False,
        help="Create a blanked output image. " "[Only works if cores=1].",
    )
    group3.add_argument(
        "--colprefix",
        dest="column_prefix",
        default=None,
        type=str,
        help='Prepend each column name with "prefix_". ' "[Default = prepend nothing]",
    )

    # SF config options
    group4 = parser.add_argument_group("Source finding/fitting configuration options")
    group4.add_argument(
        "--maxsummits",
        dest="max_summits",
        type=float,
        default=None,
        help="If more than *maxsummits* summits are detected "
        "in an island, no fitting is done, "
        "only estimation. [default: no limit]",
    )
    group4.add_argument(
        "--seedclip",
        dest="innerclip",
        type=float,
        default=5,
        help="The clipping value (in sigmas) for seeding " "islands. [default: 5]",
    )
    group4.add_argument(
        "--floodclip",
        dest="outerclip",
        type=float,
        default=4,
        help="The clipping value (in sigmas) for growing " "islands. [default: 4]",
    )
    group4.add_argument(
        "--island",
        dest="doislandflux",
        action="store_true",
        default=False,
        help="Also calculate the island flux in addition to "
        "the individual components. [default: false]",
    )
    group4.add_argument(
        "--nopositive",
        dest="nopositive",
        action="store_true",
        default=False,
        help="Don't report sources with positive fluxes. " "[default: false]",
    )
    group4.add_argument(
        "--negative",
        dest="negative",
        action="store_true",
        default=False,
        help="Report sources with negative fluxes. " "[default: false]",
    )
    group4.add_argument(
        "--region",
        dest="region",
        type=str,
        default=None,
        help="Use this regions file to restrict source finding"
        " in this image.\nUse MIMAS region (.mim) files.",
    )
    group4.add_argument(
        "--nocov",
        dest="docov",
        action="store_false",
        default=True,
        help="Don't use the covariance of the data in the "
        "fitting proccess. [Default = False]",
    )

    # priorized fitting
    group5 = parser.add_argument_group(
        "Priorized Fitting config options",
        "in addition to the above source " "fitting options",
    )
    group5.add_argument(
        "--priorized",
        dest="priorized",
        default=0,
        type=int,
        help="Enable priorized fitting level n=[1,2,3]. "
        "1=fit flux, 2=fit flux/position, "
        "3=fit flux/position/shape. "
        "See the GitHub wiki for more details.",
    )
    group5.add_argument(
        "--ratio",
        dest="ratio",
        default=None,
        type=float,
        help="The ratio of synthesized beam sizes "
        "(image psf / input catalog psf). "
        "For use with priorized.",
    )
    group5.add_argument(
        "--noregroup",
        dest="regroup",
        default=True,
        action="store_false",
        help="Do not regroup islands before priorized fitting",
    )
    group5.add_argument(
        "--input",
        dest="input",
        type=str,
        default=None,
        help="If --priorized is used, this gives the filename "
        "for a catalog of locations at which "
        "fluxes will be measured.",
    )
    group5.add_argument(
        "--catpsf",
        dest="catpsf",
        type=str,
        default=None,
        help="A psf map corresponding to the input catalog. "
        "This will allow for the correct resizing of"
        " sources when the catalog and image psfs differ",
    )
    group5.add_argument(
        "--regroup-eps",
        dest="regroup_eps",
        default=None,
        type=float,
        help="The size in arcminutes that is used to regroup "
        "nearby components into a single set of "
        "components that will be solved for "
        "simultaneously",
    )

    # Debug and extras
    group6 = parser.add_argument_group("Extra options")
    group6.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=False,
        help="Enable the saving of the background and noise "
        "images. Sets --find to false. "
        "[default: false]",
    )
    group6.add_argument(
        "--outbase",
        dest="outbase",
        type=str,
        default=None,
        help="If --save is True, then this specifies the base "
        "name of the background and noise images. "
        "[default: inferred from input image]",
    )
    group6.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Enable debug mode. [default: false]",
    )
    group6.add_argument(
        "--versions",
        dest="file_versions",
        action="store_true",
        default=False,
        help="Show the file versions of relevant modules. " "[default: false]",
    )
    group6.add_argument(
        "--cite",
        dest="cite",
        action="store_true",
        default=False,
        help="Show citation information.",
    )

    options = parser.parse_args()

    invocation_string = " ".join(sys.argv)
    print(sys.argv)
    # configure logging
    logging_level = logging.DEBUG if options.debug else logging.INFO
    logger.setLevel(logging_level)
    logger.info("This is Aegean {0}-({1})".format(__version__, __date__))

    logger.debug("Run as:\n{0}".format(invocation_string))

    # options that don't require image inputs
    if options.cite:
        print(__citation__)
        return 0

    import AegeanTools #! This import is not at the top
    from AegeanTools.source_finder import SourceFinder #! This import is not at the top

    # source finding object
    sf = SourceFinder() #! <--- This is the source finder object

    if options.table_formats:
        show_formats()
        return 0

    if options.file_versions:
        logger.info(
            "AegeanTools {0} from {1}".format(
                AegeanTools.__version__, AegeanTools.__file__
            )
        )
        logger.info("Numpy {0} from {1} ".format(np.__version__, np.__file__))
        logger.info("Scipy {0} from {1}".format(scipy.__version__, scipy.__file__))
        logger.info(
            "AstroPy {0} from {1}".format(astropy.__version__, astropy.__file__)
        )
        logger.info("LMFit {0} from {1}".format(lmfit.__version__, lmfit.__file__))
        return 0

    # print help if the user enters no options or filename
    if options.image is None:
        parser.print_help()
        return 0

    # check that a valid filename was entered
    filename = options.image
    if not os.path.exists(filename):
        logger.error("{0} not found".format(filename))
        return 1

    # check to see if the user has supplied --telescope/--psf when required
    check_projection(filename, options)

    # tell numpy to shut up about "invalid values encountered"
    # Its just NaN's and I don't need to hear about it once per core
    np.seterr(invalid="ignore", divide="ignore")

    # check for nopositive/negative conflict
    if options.nopositive and not options.negative:
        logger.warning(
            "Requested no positive sources, but no negative sources. "
            "Nothing to find."
        )
        return 0

    # if priorized/save are enabled we turn off "find" unless it was
    # specifically set
    if (options.save or options.priorized) and not options.find:
        options.find = False
    else:
        options.find = True

    # debugging in multi core mode is very hard to understand
    if options.debug:
        logger.info("Setting cores=1 for debugging")
        options.cores = 1

    # check/set cores to use
    if options.cores is None:
        options.cores = multiprocessing.cpu_count()
        logger.info("Found {0} cores".format(options.cores))
    logger.info("Using {0} cores".format(options.cores))

    hdu_index = options.hdu_index
    if hdu_index > 0:
        logger.info("Using hdu index {0}".format(hdu_index))

    # create a beam object from user input
    if options.beam is not None:
        beam = options.beam
        options.beam = Beam(beam[0], beam[1], beam[2])
        logger.info("Using user supplied beam parameters")
        logger.info(
            "Beam is {0} deg x {1} deg with pa {2}".format(
                options.beam.a, options.beam.b, options.beam.pa
            )
        )

    # auto-load background, noise, psf and region files
    basename = os.path.splitext(filename)[0]
    if options.autoload:
        files = get_aux_files(filename)
        if files["bkg"] and not options.backgroundimg:
            options.backgroundimg = files["bkg"]
            logger.info("Found background {0}".format(options.backgroundimg))
        if files["rms"] and not options.noiseimg:
            options.noiseimg = files["rms"]
            logger.info("Found noise {0}".format(options.noiseimg))
        if files["mask"] and not options.region:
            options.region = files["mask"]
            logger.info("Found region {0}".format(options.region))
        if files["psf"] and not options.imgpsf:
            options.imgpsf = files["psf"]
            logger.info("Found psf {0}".format(options.imgpsf))

    # check that the aux input files exist
    if options.backgroundimg and not os.path.exists(options.backgroundimg):
        logger.error("{0} not found".format(options.backgroundimg))
        return 1
    if options.noiseimg and not os.path.exists(options.noiseimg):
        logger.error("{0} not found".format(options.noiseimg))
        return 1
    if options.imgpsf and not os.path.exists(options.imgpsf):
        logger.error("{0} not found".format(options.imgpsf))
        return 1
    if options.catpsf and not os.path.exists(options.catpsf):
        logger.error("{0} not found".format(options.catpsf))
        return 1

    if options.region is not None:
        if not os.path.exists(options.region):
            logger.error("Region file {0} not found".format(options.region))
            return 1

    # Generate and save the background FITS files with the Aegean default
    # calculator
    if options.save:
        sf.save_background_files(
            filename,
            hdu_index=hdu_index,
            cores=options.cores,
            beam=options.beam,
            outbase=options.outbase,
            bkgin=options.backgroundimg,
            rmsin=options.noiseimg,
            rms=options.rms,
            bkg=options.bkg,
            cube_index=options.slice,
        )
        return 0

    # check that the output table formats are supported (if given)
    # BEFORE any cpu intensive work is done
    if options.tables is not None:
        if not check_table_formats(options.tables):
            logger.critical(
                "One or more output table formats are not supported: Exiting"
            )
            return 1

    # if an outputfile was specified open it for writing
    if options.outfile == "stdout":
        options.outfile = sys.stdout
    elif options.outfile is not None:
        options.outfile = open(options.outfile, "w")

    sources = []

    if options.find:
        logger.info("Finding sources.")
        found = sf.find_sources_in_image(
            filename,
            outfile=options.outfile,
            hdu_index=options.hdu_index,
            rms=options.rms,
            bkg=options.bkg,
            max_summits=options.max_summits,
            innerclip=options.innerclip,
            outerclip=options.outerclip,
            cores=options.cores,
            rmsin=options.noiseimg,
            bkgin=options.backgroundimg,
            beam=options.beam,
            doislandflux=options.doislandflux,
            nonegative=not options.negative,
            nopositive=options.nopositive,
            mask=options.region,
            imgpsf=options.imgpsf,
            blank=options.blank,
            docov=options.docov,
            cube_index=options.slice,
            progress=options.progress,
            threeD=options.threeD,
        )
        if options.blank:
            outname = basename + "_blank.fits"
            sf.save_image(outname)
        if len(found) == 0:
            logger.info("No sources found in image")

    if options.priorized > 0:
        if options.ratio is not None:
            if options.ratio <= 0:
                logger.error("ratio must be positive definite")
                return 1
            if options.ratio < 1:
                logger.error("ratio <1 is not advised. Have fun!")
        if options.input is None:
            logger.error("Must specify input catalog when " "--priorized is selected")
            return 1
        if not os.path.exists(options.input):
            logger.error("{0} not found".format(options.input))
            return 1
        logger.info("Priorized fitting of sources in input catalog.")

        logger.info("Stage = {0}".format(options.priorized))
        if options.doislandflux:
            logger.warning(
                "--island requested but not yet supported for " "priorized fitting"
            )
        sf.priorized_fit_islands(
            filename,
            catalogue=options.input,
            hdu_index=options.hdu_index,
            rms=options.rms,
            bkg=options.bkg,
            outfile=options.outfile,
            bkgin=options.backgroundimg,
            rmsin=options.noiseimg,
            beam=options.beam,
            imgpsf=options.imgpsf,
            catpsf=options.catpsf,
            stage=options.priorized,
            ratio=options.ratio,
            outerclip=options.outerclip,
            cores=options.cores,
            doregroup=options.regroup,
            docov=options.docov,
            cube_index=options.slice,
            progress=options.progress,
            regroup_eps=options.regroup_eps,
        )

    sources = sf.sources

    logger.info("found {0} sources total".format(len(sources)))
    if len(sources) > 0 and options.tables:
        meta = {
            "PROGRAM": "Aegean",
            "PROGVER": "{0}-({1})".format(__version__, __date__),
            "FITSFILE": filename,
            "RUN-AS": invocation_string,
        }

        # collect catalogues and clean up
        if MPI_AVAIL:
            logger.info("MPI is available")
            catalog_list = []
        for t in options.tables.split(","):
            final_file_name = t
            if MPI_AVAIL:
                t = addSuffix(t,MPI.COMM_WORLD.Get_rank())
            save_catalog(t, sources, prefix=options.column_prefix, meta=meta)

        if MPI_AVAIL:
            MPI.COMM_WORLD.Barrier()
            if MPI.COMM_WORLD.Get_rank() == 0:
                for t in options.tables.split(","):
                    flist = []
                    final_file_name = addSuffix(t,"comp")
                    for n in range(0, MPI.COMM_WORLD.Get_size()):
                        base = addSuffix(t,n)
                        base = addSuffix(base, "comp")
                        flist.append(base)
                    final_table = load_table(flist[0])
                    for f in flist[1:]:
                        table = load_table(f)
                        final_table = vstack([final_table, table])
                        os.remove(f)
            
                    write_table(final_table, final_file_name)
                    logger.info(f"Wrote table name: {final_file_name}")

    return 0
