#! /usr/bin/env python
"""
The Aegean source finding program.

"""

import sys
import os
import numpy as np
import scipy
import lmfit
import astropy
import logging
import logging.config

from optparse import OptionParser

# need Region in the name space in order to be able to unpickle it
try:
    from AegeanTools.regions import Region

    region_available = True
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
except ImportError:
    region_available = False


from AegeanTools.source_finder import scope2lat
from AegeanTools.fits_image import Beam
from AegeanTools.catalogs import show_formats, check_table_formats, save_catalog
import AegeanTools.pprocess as pprocess
import multiprocessing


from AegeanTools import __version__, __date__
header = """#Aegean version {0}
# on dataset: {1}"""

# Nifty helpers
# def gmean(indata):
#     """
#     Calculate the geometric mean and variance of a data set.
#
#     This function is designed such that gmean(data + a) = gmean(data) + a
#     which means that it can operate on negative values
#
#     np.nan values are excluded from the calculation however np.inf values cause the results to be also np.inf
#
#     This is the function that you are looking for when asking, what is the mean/variance of the ratio x/y, or any other
#     log-normal distributed data.
#
#     :param data: an array of numbers
#     :return: the geometric mean and variance of the data
#     """
#     # TODO: Figure out the mathematical name for functions that obey - gmean(data + a) = gmean(data) + a
#     data = np.ravel(indata)
#     if np.inf in data:
#         return np.inf, np.inf
#
#     finite = data[np.isfinite(data)]
#     if len(finite) < 1:
#         return np.nan, np.nan
#     # determine the zero point and scale all values to be 1 or greater
#     scale = abs(np.min(finite)) + 1
#     finite += scale
#     # calculate the geometric mean of the scaled data and scale back
#     lfinite = np.log(finite)
#     flux = np.exp(np.mean(lfinite)) - scale
#     error = np.nanstd(lfinite) * flux
#     return flux, abs(error)


# Functions that are not going to be used in V2.0
# def force_measure_flux(radec):
#     """
#     Measure the flux of a point source at each of the specified locations
#     Not fitting is done, just forced measurements
#     Assumes that global_data has been populated
#
#     :param radec: the locations at which to measure fluxes
#     :return: [(flux,err),...] corresponding to each ra/dec
#     """
#     # TODO: allow for a psf image to be used to make this consistent with the priorized fitting
#     from AegeanTools.fitting import ntwodgaussian_mpfit
#     catalog = []
#
#     dummy = SimpleSource()
#     dummy.peak_flux = np.nan
#     dummy.peak_pixel = np.nan
#     dummy.flags = flags.FITERR
#
#     shape = global_data.data_pix.shape
#
#     if global_data.wcshelper.lat is not None:
#         log.warn("No account is being made for telescope latitude, even though it has been supplied")
#     for ra, dec in radec:
#         # find the right pixels from the ra/dec
#         source_x, source_y = global_data.wcshelper.sky2pix([ra, dec])
#         x = int(round(source_x))
#         y = int(round(source_y))
#
#         # reject sources that are outside the image bounds, or which have nan data/rms values
#         if not 0 <= x < shape[0] or not 0 <= y < shape[1] or \
#                 not np.isfinite(global_data.data_pix[x, y]) or \
#                 not np.isfinite(global_data.rmsimg[x, y]):
#             catalog.append(dummy)
#             continue
#
#         flag = 0
#         # make a pixbeam at this location
#         pixbeam = global_data.psfhelper.get_pixbeam(ra,dec)
#         if pixbeam is None:
#             flag |= flags.WCSERR
#             pixbeam = Beam(1, 1, 0)
#         # determine the x and y extent of the beam
#         xwidth = 2 * pixbeam.a * pixbeam.b
#         xwidth /= np.hypot(pixbeam.b * np.sin(np.radians(pixbeam.pa)), pixbeam.a * np.cos(np.radians(pixbeam.pa)))
#         ywidth = 2 * pixbeam.a * pixbeam.b
#         ywidth /= np.hypot(pixbeam.b * np.cos(np.radians(pixbeam.pa)), pixbeam.a * np.sin(np.radians(pixbeam.pa)))
#         # round to an int and add 1
#         ywidth = int(round(ywidth)) + 1
#         xwidth = int(round(xwidth)) + 1
#
#         # cut out an image of this size
#         xmin = max(0, x - xwidth / 2)
#         ymin = max(0, y - ywidth / 2)
#         xmax = min(shape[0], x + xwidth / 2 + 1)
#         ymax = min(shape[1], y + ywidth / 2 + 1)
#         data = global_data.data_pix[xmin:xmax, ymin:ymax]
#
#         # Make a Gaussian equal to the beam with amplitude 1.0 at the position of the source
#         # in terms of the pixel region.
#         amp = 1.0
#         xo = source_x - xmin
#         yo = source_y - ymin
#         params = [amp, xo, yo, pixbeam.a * FWHM2CC, pixbeam.b * FWHM2CC, pixbeam.pa]
#         gaussian_data = ntwodgaussian_mpfit(params)(*np.indices(data.shape))
#
#         # Calculate the "best fit" amplitude as the average of the implied amplitude
#         # for each pixel. Error is stddev.
#         # Only use pixels within the FWHM, ie value>=0.5. Set the others to NaN
#         ratios = np.where(gaussian_data >= 0.5, data / gaussian_data, np.nan)
#         flux, error = gmean(ratios)
#
#         # sources with fluxes or flux errors that are not finite are not valid
#         # an error of identically zero is also not valid.
#         if not np.isfinite(flux) or not np.isfinite(error) or error == 0.0:
#             catalog.append(dummy)
#             continue
#
#         source = SimpleSource()
#         source.ra = ra
#         source.dec = dec
#         source.peak_flux = flux
#         source.err_peak_flux = error
#         source.background = global_data.bkgimg[x, y]
#         source.flags = flag
#         source.peak_pixel = np.nanmax(data)
#         source.local_rms = global_data.rmsimg[x, y]
#         source.a = global_data.beam.a
#         source.b = global_data.beam.b
#         source.pa = global_data.beam.pa
#
#         catalog.append(source)
#         if logging.getLogger('Aegean').isEnabledFor(logging.DEBUG):
#             log.debug("Measured source {0}".format(source))
#             log.debug("  used area = [{0}:{1},{2}:{3}]".format(xmin, xmax, ymin, ymax))
#             log.debug("  xo,yo = {0},{1}".format(xo, yo))
#             log.debug("  params = {0}".format(params))
#             log.debug("  flux at [xmin+xo,ymin+yo] = {0} Jy".format(data[int(xo), int(yo)]))
#             log.debug("  error = {0}".format(error))
#             log.debug("  rms = {0}".format(source.local_rms))
#     return catalog
#
#
# def measure_catalog_fluxes(filename, catfile, hdu_index=0, outfile=None, bkgin=None, rmsin=None, cores=1, rms=None,
#                            beam=None, lat=None):
#     """
#     Measure the flux at a given set of locations, assuming point sources.
#
#     This function is of limited use, priorized_fit_islands should be used instead.
#
#     :param filename: filename or HDUList of image
#     :param catfile: a catalog of source positions (ra,dec)
#     :param hdu_index: if fits file has more than one hdu, it can be specified here
#     :param outfile: the output file to write to (NOT a table)
#     :param bkgin: a background image filename or HDUList
#     :param rmsin: an rms image filename or HDUList
#     :param cores: cores to use
#     :param rms: forced rms value
#     :param beam: beam parameters to override those given in fits header
#     :param lat: telescope latitude (ignored)
#     :return: a list of simple sources
#     """
#
#     load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, verb=True,
#                  do_curve=False, beam=beam, lat=lat)
#
#     # load catalog
#     radec = load_catalog(catfile)
#     # measure fluxes
#     sources = force_measure_flux(radec)
#     # write output
#     print >> outfile, header.format("{0}-({1})".format(__version__,__date__), filename)
#     print >> outfile, SimpleSource.header
#     for source in sources:
#         print >> outfile, str(source)
#     return sources
#
#
# def VASTP_measure_catalog_fluxes(filename, positions, hdu_index=0, bkgin=None, rmsin=None,
#                                  rms=None, cores=1, beam=None, debug=False):
#     """
#     A version of measure_catalog_fluxes that will accept a list of positions instead of reading from a file.
#     Input:
#         filename - fits image file name to be read
#         positions - a list of source positions (ra,dec)
#         hdu_index - if fits file has more than one hdu, it can be specified here
#         outfile - the output file to write to
#         bkgin - a background image filename
#         rmsin - an rms image filename
#         cores - cores to use
#         rms - forced rms value
#         beam - beam parameters to override those given in fits header
#     """
#     load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, beam=beam, verb=True,
#                  do_curve=False)
#     # measure fluxes
#     if debug:
#         level = log.getLogger().getEffectiveLevel()
#         log.getLogger().setLevel(log.DEBUG)
#     sources = force_measure_flux(positions)
#     if debug:
#         log.getLogger().setLevel(level)
#     return sources
#
#
# def VASTP_refit_sources(filename, sources, hdu_index=0, bkgin=None, rmsin=None, rms=None, cores=1, beam=None, debug=False):
#     """
#     A version of priorized_fit_islands that will work with the vast pipeline
#     Input:
#         filename - fits image file name to be read
#         sources - a list of source objects
#         hdu_index - if fits file has more than one hdu, it can be specified here
#         outfile - the output file to write to
#         bkgin - a background image filename
#         rmsin - an rms image filename
#         cores - cores to use
#         rms - forced rms value
#         beam - beam parameters to override those given in fits header
#     """
#     logging.info(" refitting {0} sources".format(len(sources)))
#     if len(sources)<1:
#         return []
#     stage = 1
#     outerclip = 4 # ultimately ignored but required for now
#
#     load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, cores=cores, verb=True,
#                  do_curve=False, beam=beam)
#
#     new_sources = []
#
#     for src in sources:
#         res = refit_islands([[src]], stage, outerclip)
#         # if the source is not able to be fit then we dummy the source
#         if len(res)<1:
#             d = OutputSource()
#             d.peak_flux = np.nan
#             res = [d]
#         elif len(res)!=2:
#             logging.error("expecting two sources, but got {0}".format(len(res)))
#         s = res[0]
#         s.flags = 0
#         # convert a/b from deg->arcsec to emulate a mistake that was made in forcemeasurements
#         # and then corrected in the vast pipeline (ugh).
#         s.a /=3600
#         s.b /=3600
#
#         new_sources.append(s)
#
#     logging.info("Returning {0} sources".format(len(new_sources)))
#     return new_sources
#

def save_background_files(image_filename, hdu_index=0, bkgin=None, rmsin=None, beam=None, rms=None, cores=1,
                          outbase=None):
    """
    Generate and save the background and RMS maps as FITS files.
    They are saved in the current directly as aegean-background.fits and aegean-rms.fits.

    :param image_filename: filename or HDUList of image
    :param hdu_index: if fits file has more than one hdu, it can be specified here
    :param bkgin: a background image filename or HDUList
    :param rmsin: an rms image filename or HDUList
    :param beam: beam parameters to override those given in fits header
    :param rms: forced rms value
    :param cores: cores to use
    :param outbase: basename for output files
    :return:
    """
    global global_data

    log.info("Saving background / RMS maps")
    # load image, and load/create background/rms images
    load_globals(image_filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, beam=beam, verb=True, rms=rms,
                 cores=cores, do_curve=True)
    img = global_data.img
    bkgimg, rmsimg = global_data.bkgimg, global_data.rmsimg
    curve = np.array(global_data.dcurve,dtype=np.float32)
    # mask these arrays have the same mask the same as the data
    mask = np.where(np.isnan(global_data.data_pix))
    bkgimg[mask] = np.NaN
    rmsimg[mask] = np.NaN
    curve[mask] = np.NaN

    # Generate the new FITS files by copying the existing HDU and assigning new data.
    # This gives the new files the same WCS projection and other header fields.
    new_hdu = img.hdu
    # Set the ORIGIN to indicate Aegean made this file
    new_hdu.header["ORIGIN"] = "Aegean {0}-({1})".format(__version__,__date__)
    for c in ['CRPIX3', 'CRPIX4', 'CDELT3', 'CDELT4', 'CRVAL3', 'CRVAL4', 'CTYPE3', 'CTYPE4']:
        if c in new_hdu.header:
            del new_hdu.header[c]

    if outbase is None:
        outbase, _ = os.path.splitext(os.path.basename(image_filename))
    noise_out = outbase + '_rms.fits'
    background_out = outbase + '_bkg.fits'
    curve_out = outbase +'_crv.fits'

    new_hdu.data = bkgimg
    new_hdu.writeto(background_out, clobber=True)
    log.info("Wrote {0}".format(background_out))

    new_hdu.data = rmsimg
    new_hdu.writeto(noise_out, clobber=True)
    log.info("Wrote {0}".format(noise_out))

    new_hdu.data = curve
    new_hdu.writeto(curve_out, clobber = True)
    log.info("Wrote {0}".format(curve_out))
    return


if __name__ == "__main__":
    usage = "usage: %prog [options] FileName.fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--find", dest='find', action='store_true', default=False,
                      help='Source finding mode. [default: true, unless --save or --measure are selected]')
    parser.add_option("--cores", dest="cores", type="int", default=None,
                      help="Number of CPU cores to use for processing [default: all cores]")
    parser.add_option("--debug", dest="debug", action="store_true", default=False,
                      help="Enable debug mode. [default: false]")
    parser.add_option("--hdu", dest="hdu_index", type="int", default=0,
                      help="HDU index (0-based) for cubes with multiple images in extensions. [default: 0]")
    parser.add_option("--out", dest='outfile', default=None,
                      help="Destination of Aegean catalog output. [default: stdout]")
    parser.add_option("--table", dest='tables', default=None,
                      help="Additional table outputs, format inferred from extension. [default: none]")
    parser.add_option("--tformats",dest='table_formats', action="store_true",default=False,
                      help='Show a list of table formats supported in this install, and their extensions')
    parser.add_option("--forcerms", dest='rms', type='float', default=None,
                      help="Assume a single image noise of rms, and a background of zero. [default: false]")
    parser.add_option("--noise", dest='noiseimg', default=None,
                      help="A .fits file that represents the image noise (rms), created from Aegean with --save " +
                           "or BANE. [default: none]")
    parser.add_option('--background', dest='backgroundimg', default=None,
                      help="A .fits file that represents the background level, created from Aegean with --save " +
                           "or BANE. [default: none]")
    parser.add_option('--psf', dest='imgpsf',default=None,
                      help="A .fits file that represents the size (degrees) of a blurring disk. " +
                           "This disk is convolved with the BMAJ/BMIN listed in the FITS header and " +
                           "the result becomes the local PSF.")

    parser.add_option('--autoload', dest='autoload', action="store_true", default=False,
                      help="Automatically look for background, noise, region, and psf files using the input filename as a hint. [default: don't do this]")
    parser.add_option("--maxsummits", dest='max_summits', type='float', default=None,
                      help="If more than *maxsummits* summits are detected in an island, no fitting is done, only estimation. [default: no limit]")
    parser.add_option('--seedclip', dest='innerclip', type='float', default=5,
                      help='The clipping value (in sigmas) for seeding islands. [default: 5]')
    parser.add_option('--floodclip', dest='outerclip', type='float', default=4,
                      help='The clipping value (in sigmas) for growing islands. [default: 4]')
    parser.add_option('--beam', dest='beam', type='float', nargs=3, default=None,
                      help='The beam parameters to be used is "--beam major minor pa" all in degrees. [default: read from fits header].')
    parser.add_option('--telescope', dest='telescope', type=str, default=None,
                      help='The name of the telescope used to collect data. [MWA|VLA|ATCA|LOFAR]')
    parser.add_option('--lat', dest='lat', type=float, default=None,
                      help='The latitude of the telescope used to collect data.')
    parser.add_option('--versions', dest='file_versions', action="store_true", default=False,
                      help='Show the file versions of relevant modules. [default: false]')
    parser.add_option('--island', dest='doislandflux', action="store_true", default=False,
                      help='Also calculate the island flux in addition to the individual components. [default: false]')
    parser.add_option('--nopositive', dest='nopositive', action="store_true", default=False,
                      help="Don't report sources with positive fluxes. [default: false]")
    parser.add_option('--negative', dest='negative', action="store_true", default=False,
                      help="Report sources with negative fluxes. [default: false]")

    parser.add_option('--region', dest='region', default=None,
                      help="Use this regions file to restrict source finding in this image.")

    parser.add_option('--save', dest='save', action="store_true", default=False,
                      help='Enable the saving of the background and noise images. Sets --find to false. [default: false]')
    parser.add_option('--outbase', dest='outbase', default=None,
                      help='If --save is True, then this specifies the base name of the background and noise images. [default: inferred from input image]')

    parser.add_option('--measure', dest='measure', action='store_true', default=False,
                      help='Enable forced measurement mode. Requires an input source list via --input. Sets --find to false. [default: false]')
    parser.add_option('--priorized', dest='priorized', default=0, type=int,
                      help="Enable priorized fitting, with stage = n [default=1]")
    parser.add_option('--ratio', dest='ratio', default=None, type=float,
                      help="The ratio of synthesized beam sizes (image psf / input catalog psf). For use with priorized.")
    parser.add_option('--noregroup',dest='regroup', default=True, action='store_false',
                      help='Do not regroup islands before priorized fitting.')
    parser.add_option('--input', dest='input', default=None,
                      help='If --measure is true, this gives the filename for a catalog of locations at which fluxes will be measured. [default: none]')
    parser.add_option('--catpsf', dest='catpsf', default=None,
                      help='A psf map corresponding to the input catalog. This will allow for the correct resizing of '+
                           'sources when the catalog and image psfs differ.')

    (options, args) = parser.parse_args()

    # configure logging
    logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
    log = logging.getLogger("Aegean")
    logging_level = logging.DEBUG if options.debug else logging.INFO
    log.setLevel(logging_level)
    log.info("This is Aegean {0}-({1})".format(__version__,__date__))

    from AegeanTools.source_finder import SourceFinder, check_cores
    # source finding object
    sf = SourceFinder(log=log)

    if options.table_formats:
        show_formats()
        sys.exit(0)

    if options.file_versions:
        log.info("Numpy {0} from {1} ".format(np.__version__, np.__file__))
        log.info("Scipy {0} from {1}".format(scipy.__version__, scipy.__file__))
        log.info("AstroPy {0} from {1}".format(astropy.__version__, astropy.__file__))
        log.info("LMFit {0} from {1}".format(lmfit.__version__, lmfit.__file__))
        try:
            import h5py
            log.info("h5py {0} from {1}".format(h5py.__version__, h5py.__file__))
        except ImportError:
            log.info("h5py not found")
        sys.exit(0)

    # print help if the user enters no options or filename
    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    # check that a valid filename was entered
    filename = args[0]
    if not os.path.exists(filename):
        log.error("{0} not found".format(filename))
        sys.exit(1)

    # tell numpy to shut up about "invalid values encountered"
    # Its just NaN's and I don't need to hear about it once per core
    np.seterr(invalid='ignore', divide='ignore')

    # check for nopositive/negative conflict
    if options.nopositive and not options.negative:
        log.warning('Requested no positive sources, but no negative sources. Nothing to find.')
        sys.exit()

    # if measure/save are enabled we turn off "find" unless it was specifically set
    if (options.measure or options.save or options.priorized) and not options.find:
        options.find = False
    else:
        options.find = True

    # debugging in multi core mode is very hard to understand
    if options.debug:
        log.info("Setting cores=1 for debugging")
        options.cores = 1

    # check/set cores to use
    if options.cores is None:
        options.cores = multiprocessing.cpu_count()
        log.info("Found {0} cores".format(options.cores))
    if options.cores > 1:
        options.cores = check_cores(options.cores)
    log.info("Using {0} cores".format(options.cores))

    hdu_index = options.hdu_index
    if hdu_index > 0:
        log.info("Using hdu index {0}".format(hdu_index))

    # create a beam object from user input
    if options.beam is not None:
        beam = options.beam
        if len(beam) != 3:
            beam = beam.split()
            print "Beam requires 3 args. You supplied '{0}'".format(beam)
            sys.exit(1)
        options.beam = Beam(beam[0], beam[1], beam[2])
        log.info("Using user supplied beam parameters")
        log.info("Beam is {0} deg x {1} deg with pa {2}".format(options.beam.a, options.beam.b, options.beam.pa))

    # determine the latitude of the telescope
    if options.telescope is not None:
        lat = scope2lat(options.telescope)
    elif options.lat is not None:
        lat = options.lat
    else:
        lat = None

    # Generate and save the background FITS files
    if options.save:
        raise NotImplementedError("check back later")
        # save_background_files(filename, hdu_index=hdu_index, cores=options.cores, beam=options.beam,
        #                       outbase=options.outbase)

    # auto-load background, noise, psf and region files
    if options.autoload:
        basename = os.path.splitext(filename)[0]
        if os.path.exists(basename+'_bkg.fits'):
            options.backgroundimg = basename+'_bkg.fits'
            log.info("Found background {0}".format(options.backgroundimg))
        if os.path.exists(basename+"_rms.fits"):
            options.noiseimg = basename+'_rms.fits'
            log.info("Found noise {0}".format(options.noiseimg))
        if os.path.exists(basename+".mim"):
            options.region = basename+".mim"
            log.info("Found region {0}".format(options.region))
        if os.path.exists(basename+"_psf.fits"):
            options.imgpsf = basename +"_psf.fits"
            log.info("Found psf {0}".format(options.imgpsf))

    # check that the aux input files exist
    if options.backgroundimg and not os.path.exists(options.backgroundimg):
        log.error("{0} not found".format(options.backgroundimg))
        sys.exit(1)
    if options.noiseimg and not os.path.exists(options.noiseimg):
        log.error("{0} not found".format(options.noiseimg))
        sys.exit(1)
    if options.imgpsf and not os.path.exists(options.imgpsf):
        log.error("{0} not found".format(options.imgpsf))
        sys.exit(1)
    if options.catpsf and not os.path.exists(options.catpsf):
        log.error("{0} not found".format(options.catpsf))
        sys.exit(1)

    if options.region is not None:
        if not os.path.exists(options.region):
            log.error("Region file {0} not found")
            sys.exit(1)
        if not region_available:
            log.error("Could not import AegeanTools/Region.py")
            log.error("(you probably need to install HealPy)")
            sys.exit(1)

    # check that the output table formats are supported (if given)
    # BEFORE any cpu intensive work is done
    if options.tables is not None:
        check_table_formats(options.tables)

    # if an outputfile was specified open it for writing, otherwise use stdout
    if not options.outfile:
        options.outfile = sys.stdout
    else:
        options.outfile = open(options.outfile, 'w')

    sources = []

    # do forced measurements using catfile
    if options.measure and options.priorized==0:
        raise NotImplementedError("forced measurements are not supported")
        # if options.input is None:
        #     log.error("Must specify input catalog when --measure is selected")
        #     sys.exit(1)
        # if not os.path.exists(options.input):
        #     log.error("{0} not found".format(options.input))
        #     sys.exit(1)
        # log.info("Measuring fluxes of input catalog.")
        # measurements = measure_catalog_fluxes(filename, catfile=options.input, hdu_index=options.hdu_index,
        #                                       outfile=options.outfile, bkgin=options.backgroundimg,
        #                                       rmsin=options.noiseimg, beam=options.beam, lat=lat)
        # if len(measurements) == 0:
        #     log.info("No measurements made")
        # sources.extend(measurements)

    if options.priorized>0:
        if options.ratio is not None:
            if options.ratio<=0:
                log.error("ratio must be positive definite")
                sys.exit(1)
            if options.ratio<1:
                log.error("ratio <1 is not advised. Have fun!")
        if options.input is None:
            log.error("Must specify input catalog when --priorized is selected")
            sys.exit(1)
        if not os.path.exists(options.input):
            log.error("{0} not found".format(options.input))
            sys.exit(1)
        log.info("Priorized fitting of sources in input catalog.")

        log.info("Stage = {0}".format(options.priorized))
        if options.doislandflux:
            log.warn("--island requested but not yet supported for priorized fitting")
        sf.priorized_fit_islands(filename, catfile=options.input, hdu_index=options.hdu_index,
                                            rms=options.rms,
                                            outfile=options.outfile, bkgin=options.backgroundimg,
                                            rmsin=options.noiseimg, beam=options.beam, lat=lat, imgpsf=options.imgpsf,
                                            catpsf=options.catpsf,
                                            stage=options.priorized, ratio=options.ratio, outerclip=options.outerclip,
                                            cores=options.cores, doregroup=options.regroup)

    if options.find:
        log.info("Finding sources.")
        found = sf.find_sources_in_image(filename, outfile=options.outfile, hdu_index=options.hdu_index,
                                           rms=options.rms,
                                           max_summits=options.max_summits,
                                           innerclip=options.innerclip,
                                           outerclip=options.outerclip, cores=options.cores, rmsin=options.noiseimg,
                                           bkgin=options.backgroundimg, beam=options.beam,
                                           doislandflux=options.doislandflux,
                                           nonegative=not options.negative, nopositive=options.nopositive,
                                           mask=options.region, lat=lat, imgpsf=options.imgpsf)
        if len(found) == 0:
            log.info("No sources found in image")

    sources = sf.sources
    log.info("found {0} sources total".format(len(sources)))
    if len(sources) > 0 and options.tables:
        meta = {"PROGRAM":"Aegean",
                "PROGVER":"{0}-({1})".format(__version__,__date__),
                "FITSFILE":filename}
        for t in options.tables.split(','):
            save_catalog(t, sources)
    sys.exit()
