#! /usr/bin/env python

"""
Different types of sources that Aegean is able to fit

"""

__author__ = "Paul Hancock"

import numpy as np
import uuid

class SimpleSource(object):
    """
    A (forced) measurement of flux at a given location
    """
    header = "#RA           DEC          Flux      err     a     b         pa  flags\n" + \
             "#                        Jy/beam   Jy/beam   ''    ''        deg WNCPES\n" + \
             "#======================================================================="

    formatter = "{0.ra:11.7f} {0.dec:11.7f} {0.peak_flux: 8.6f} {0.err_peak_flux: 8.6f} {0.a:5.2f} {0.b:5.2f} {0.pa:6.1f} {0.flags:06b}"
    names = ['background', 'local_rms', 'ra', 'dec', 'peak_flux', 'err_peak_flux', 'flags', 'peak_pixel', 'a', 'b',
             'pa','uuid']

    def __init__(self):
        self.background = 0.0
        self.local_rms = 0.0
        self.ra = 0.0
        self.dec = 0.0
        self.peak_flux = 0.0
        self.err_peak_flux = 0.0
        self.flags = 0
        self.peak_pixel = 0.0
        self.a = 0.0
        self.b = 0.0
        self.pa = 0.0
        self.uuid = str(uuid.uuid4())

    def sanitise(self):
        """
        Convert various numpy types to np.float64 so that they will print properly
        """
        for k in self.__dict__:
            if type(self.__dict__[k]) in [np.float32]:  # np.float32 has a broken __str__ method
                self.__dict__[k] = np.float64(self.__dict__[k])

    def __str__(self):
        """
        convert to string
        """
        self.sanitise()
        return self.formatter.format(self)

    def __repr__(self):
        """
        A string representation of the name of this source
        which is always just an empty string
        """
        return self.__str__()

    def as_list(self):
        """
        return an ordered list of the source attributes
        """
        self.sanitise()
        l = []
        for name in self.names:
            l.append(getattr(self, name))
        return l


class IslandSource(SimpleSource):
    """
    Each island of pixels can be characterised in a basic manner.
    This class contains info relevant to such objects.
    """
    names = ['island', 'components', 'background', 'local_rms', 'ra_str', 'dec_str', 'ra', 'dec',
             'peak_flux', 'int_flux', 'err_int_flux', 'eta', 'x_width', 'y_width', 'max_angular_size', 'pa',
             'pixels', 'area', 'beam_area', 'flags','uuid']

    def __init__(self):
        SimpleSource.__init__(self)
        self.island = 0  # island number
        #background = None # local background zero point
        #local_rms= None #local image rms
        self.ra_str = ''  # str
        self.dec_str = ''  # str
        #ra = None # degrees
        #dec = None # degrees
        #peak_flux = None # Jy/beam
        self.int_flux = 0.0  # Jy
        self.err_int_flux = 0.0  # Jy
        self.x_width = 0
        self.y_width = 0
        self.max_angular_size = 0
        self.pa = 0
        self.pixels = 0
        self.area = 0
        self.beam_area = 0  # at the brightest pixel
        self.components = 0
        self.eta = 0.0
        # not included in 'names' and thus not included by default in most output
        self.extent = 0
        self.contour = []
        self.max_angular_size_anchors = []
        self.pix_mask = [] # the ra/dec of all the non masked pixels in this island.

    def __str__(self):
        return "({0:d})".format(self.island)

    def __cmp__(self, other):
        """
        sort order is firstly by island then by source
        both in ascending order
        """
        if self.island > other.island:
            return 1
        elif self.island < other.island:
            return -1
        else:
            return 0


class OutputSource(SimpleSource):
    """
    Each source that is fit by Aegean is cast to this type.
    The parameters of the source are stored, along with a string
    formatter that makes printing easy. (as does OutputSource.__str__)
    """
    #header for the output
    header = "#isl,src   bkg       rms         RA           DEC         RA         err         DEC        err         Peak      err     S_int     err        a    err    b    err     pa   err   flags\n" + \
             "#         Jy/beam   Jy/beam                               deg        deg         deg        deg       Jy/beam   Jy/beam    Jy       Jy         ''    ''    ''    ''    deg   deg   WNCPES\n" + \
             "#==========================================================================================================================================================================================="

    #formatting strings for making nice output
    formatter = "({0.island:04d},{0.source:02d}) {0.background: 8.6f} {0.local_rms: 8.6f} " + \
                "{0.ra_str:12s} {0.dec_str:12s} {0.ra:11.7f} {0.err_ra: 9.7f} {0.dec:11.7f} {0.err_dec: 9.7f} " + \
                "{0.peak_flux: 8.6f} {0.err_peak_flux: 8.6f} {0.int_flux: 8.6f} {0.err_int_flux: 8.6f} " + \
                "{0.a:5.2f} {0.err_a:5.2f} {0.b:5.2f} {0.err_b:5.2f} " + \
                "{0.pa:6.1f} {0.err_pa:5.1f}   {0.flags:06b}"
    names = ['island', 'source', 'background', 'local_rms', 'ra_str', 'dec_str', 'ra', 'err_ra', 'dec', 'err_dec',
             'peak_flux', 'err_peak_flux', 'int_flux', 'err_int_flux', 'a', 'err_a', 'b', 'err_b', 'pa', 'err_pa',
             'flags','residual_mean','residual_std','uuid','psf_a','psf_b','psf_pa']

    def __init__(self):
        SimpleSource.__init__(self)
        self.island = 0  # island number
        self.source = 0  # source number
        #background = None # local background zero point
        #local_rms= None #local image rms
        self.ra_str = ''  #str
        self.dec_str = ''  #str
        #ra = None # degrees
        self.err_ra = 0.0  # degrees
        #dec = None # degrees
        self.err_dec = 0.0
        #peak_flux = None # Jy/beam
        #err_peak_flux = None # Jy/beam
        self.int_flux = 0.0  #Jy
        self.err_int_flux = 0.0  #Jy
        #self.a = 0.0 # major axis (arcsecs)
        self.err_a = 0.0  # arcsecs
        #self.b = 0.0 # minor axis (arcsecs)
        self.err_b = 0.0  # arcsecs
        #self.pa = 0.0 # position angle (degrees - WHAT??)
        self.err_pa = 0.0  # degrees
        self.flags = 0x0
        self.residual_mean = 0
        self.residual_std = 0
        #
        self.psf_a = 0
        self.psf_b = 0
        self.psf_pa = 0

    def __str__(self):
        self.sanitise()
        return self.formatter.format(self)

    def as_list_dep(self):
        """Return a list of all the parameters that are stored in this Source"""
        return [self.island, self.source, self.background, self.local_rms,
                self.ra_str, self.dec_str, self.ra, self.err_ra, self.dec, self.err_dec,
                self.peak_flux, self.err_peak_flux, self.int_flux, self.err_int_flux,
                self.a, self.err_a, self.b, self.err_b,
                self.pa, self.err_pa, self.flags]

    def __repr__(self):
        return "({0:d},{1:d})".format(self.island, self.source)

    def __cmp__(self, other):
        """
        sort order is firstly by island then by source
        both in ascending order
        """
        if self.island > other.island:
            return 1
        elif self.island < other.island:
            return -1
        else:
            # if we try to compare a component to and island we put the
            if not hasattr(other,'source'):
                return -1
            try:
                if self.source > other.source:
                    return 1
                elif self.source < other.source:
                    return -1
                else:
                    return 0
            except AttributeError:
                # if we try to compare a component to and island we put the island first
                if not hasattr(other,'source'):
                    return 1


class GlobalFittingData(object):
    """
    The global data used for fitting.
    (should be) Read-only once created.
    Used by island fitting subprocesses.
    wcs parameter used by most functions.
    """

    def __init__(self):
        self.img = None
        self.dcurve = None
        self.rmsimg = None
        self.bkgimg = None
        self.hdu_header = None
        self.beam = None
        self.wcs = None
        self.data_pix = None
        self.dtype = None
        self.region = None
        self.telescope_lat = None
        return


class IslandFittingData(object):
    """
    All the data required to fit a single island.
    Instances are pickled and passed to the fitting subprocesses

    isle_num = island number (int)
    i = the pixel island (a 2D numpy array of pixel values)
    scalars=(innerclip,outerclip,max_summits)
    offsets=(xmin,xmax,ymin,ymax)
    """

    def __init__(self, isle_num=0, i=None, scalars=None, offsets=(0,0,1,1), doislandflux=False):
        self.isle_num = isle_num
        self.i = i
        self.scalars = scalars
        self.offsets = offsets
        self.doislandflux = doislandflux


def classify_catalog(catalog):
    """
    look at a catalog of sources and split them according to their class
    returns:
    components - sources of type OutputSource
    islands - sources of type IslandSource
    """
    components = []
    islands = []
    simples = []
    for source in catalog:
        if isinstance(source, OutputSource):
            components.append(source)
        elif isinstance(source, IslandSource):
            islands.append(source)
        elif isinstance(source, SimpleSource):
            simples.append(source)
    return components, islands, simples


def island_itergen(catalog):
    """
    Iterate over a catalog of sources, and return an island worth of sources at a time.
    Yields a list of components, one island at a time

    :param catalog: A list of objects which have island/source attributes
    :return:
    """
    # reverse sort so that we can pop the last elements and get an increasing island number
    catalog = sorted(catalog)
    catalog.reverse()
    group = []

    # using pop and keeping track of the list length ourselves is faster than
    # constantly asking for len(catalog)
    src = catalog.pop()
    c_len = len(catalog)
    isle_num = src.island
    while c_len >= 0:
        if src.island == isle_num:
            group.append(src)
            c_len -= 1
            if c_len <0:
                # we have just added the last item from the catalog
                # and there are no more to pop
                yield group
            else:
                src = catalog.pop()
        else:
            isle_num += 1
            # maybe there are no sources in this island so skip it
            if group == []:
                continue
            yield group
            group = []
    return