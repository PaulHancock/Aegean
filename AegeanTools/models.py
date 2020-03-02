#! /usr/bin/env python
"""
Different types of sources that Aegean is able to fit

"""

from __future__ import print_function
import numpy as np
import uuid

__author__ = "Paul Hancock"


class SimpleSource(object):
    """
    The base source class for an elliptical Gaussian.

    Attributes
    ----------
    background, local_rms : float
        Background and local noise level in the image at the location of this source.

    ra, dec : float
        Sky location of this source. Decimal degrees.

    galactic : bool
        If true then ra,dec are interpreted as glat,glon instead.
        Default = False.
        This is a class attribute, not an instance attribute.

    peak_flux, err_peak_flux : float
        The peak flux value and associated uncertainty.

    peak_pixel : float
        Value of the brightest pixel for this source.

    flags : int
        Flags. See :module:`AegeanTools.flags`.

    a, b, pa : float
        Shape parameters for this source.

    uuid : str
        Unique ID for this source. This is random and not dependent on the source properties.

    See Also
    --------
    :module:`AegeanTools.flags`
    """
    header = "#RA           DEC          Flux      err     a     b         pa     flags\n" + \
             "#                        Jy/beam   Jy/beam   ''    ''        deg  ZWNCPES\n" + \
             "#========================================================================"

    formatter = "{0.ra:11.7f} {0.dec:11.7f} {0.peak_flux: 8.6f} {0.err_peak_flux: 8.6f} {0.a:5.2f} {0.b:5.2f} {0.pa:6.1f} {0.flags:07b}"
    names = ['background', 'local_rms', 'ra', 'dec', 'peak_flux', 'err_peak_flux', 'flags', 'peak_pixel', 'a', 'b',
             'pa', 'uuid']
    galactic = False

    def __init__(self):
        self.background = np.nan
        self.local_rms = np.nan
        self.ra = np.nan
        self.dec = np.nan
        self.peak_flux = np.nan
        self.err_peak_flux = np.nan
        self.flags = 0x0
        self.peak_pixel = np.nan
        self.a = np.nan
        self.b = np.nan
        self.pa = np.nan
        self.uuid = str(uuid.uuid4())

    def _sanitise(self):
        """
        Convert attributes of type npumpy.float32 to numpy.float64 so that they will print properly.
        """
        for k in self.__dict__:
            if isinstance(self.__dict__[k], np.float32):  # np.float32 has a broken __str__ method
                self.__dict__[k] = np.float64(self.__dict__[k])

    def __str__(self):
        self._sanitise()
        return self.formatter.format(self)

    def __repr__(self):
        return self.__str__()

    def as_list(self):
        """
        Return an *ordered* list of the source attributes
        """
        self._sanitise()
        l = []
        for name in self.names:
            l.append(getattr(self, name))
        return l


class IslandSource(SimpleSource):
    """
    An island of pixels.


    Attributes
    ----------
    island: int
        The island identification number

    components : int
        The number of components that make up this island.

    background, local_rms : float
        Background and local noise level in the image at the location of this source.

    ra, dec : float
        Sky location of the brightest pixel in this island. Decimal degrees.

    ra_str, dec_str : str
        Sky location in HH:MM:SS.SS +DD:MM:SS.SS format.

    galactic : bool
        If true then ra,dec are interpreted as glat,glon instead.
        Default = False.
        This is a class attribute, not an instance attribute.

    peak_flux, peak_pixel : float
        Value of the brightest pixel for this source.

    int_flux, err_int_flux : float
        Integrated flux and associated uncertainty.

    x_width, y_width : int
        The extent of the island in pixel space. The width is of the smallest bounding box.

    max_angular_size : float
        The maximum angular size of the island in sky coordinates (degrees).

    pa : float
        Position angle for the line representing the maximum angular size.

    pixels : int
        The number of pixels covered by this island.

    area : float
        The area of this island in sky coordinates (square degrees).

    beam_area : float
        The area of the synthesized beam of the image at the location of the brightest pixel.
        (square degrees).

    eta : float
        A factor that accounts for the difference between the integrated flux
        counted by summing pixels, and the integrated flux that would be produced
        by integrating an appropriately sized Gaussian.

    extent : float

    contour : list
        A list of pixel coordinates that mark the pixel boundaries for this island
        of pixels.

    max_angular_size_anchors : [x1, y1, x2, y2]
        The end points of the vector that describes the maximum angular size
        of this island.

    flags : int
        Flags. See :module:`AegeanTools.flags`.

    uuid : str
        Unique ID for this source. This is random and not dependent on the source properties.

    See Also
    --------
    :module:`AegeanTools.flags`

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
        self.int_flux = np.nan  # Jy
        self.err_int_flux = np.nan  # Jy
        self.x_width = np.nan
        self.y_width = np.nan
        self.max_angular_size = np.nan
        self.pa = np.nan
        self.pixels = np.nan
        self.area = np.nan
        self.beam_area = np.nan  # at the brightest pixel
        self.components = np.nan
        self.eta = np.nan
        # not included in 'names' and thus not included by default in most output
        self.extent = np.nan
        self.contour = []
        self.max_angular_size_anchors = []
        self.pix_mask = [] # the ra/dec of all the non masked pixels in this island.

    def __str__(self):
        return "({0:d})".format(self.island)

    def __eq__(self, other):
        if hasattr(other, 'island'):
            return self.island == other.island
        else:
            return False

    def __ne__(self, other):
        if hasattr(other, 'island'):
            return self.island != other.island
        else:
            return True

    def __lt__(self, other):
        if hasattr(other, 'island'):
            return self.island < other.island
        else:
            return True

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        if hasattr(other, 'island'):
            return self.island > other.island
        else:
            return False

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)


class ComponentSource(SimpleSource):
    """
    A Gaussian component, aka a source, that was measured by Aegean.

    Attributes
    ----------
    island : int
        The island which this component is part of.

    source : int
        The source number within the island.

    background, local_rms : float
        Background and local noise level in the image at the location of this source.

    ra, err_ra, dec, err-dec : float
        Sky location of the source including uncertainties. Decimal degrees.

    ra_str, dec_str : str
        Sky location in HH:MM:SS.SS +DD:MM:SS.SS format.

    galactic : bool
        If true then ra,dec are interpreted as glat,glon instead.
        Default = False.
        This is a class attribute, not an instance attribute.

    peak_flux, err_peak_flux : float
        The peak flux and associated uncertainty.

    int_flux, err_int_flux : float
        Integrated flux and associated uncertainty.

    a, err_a, b, err_b, pa, err_pa: float
        Shape parameters for this source and associated uncertainties.
        a/b are in arcsec, pa is in degrees East of North.

    residual_mean, residual_std : float
        The mean and standard deviation of the model-data for this island
        of pixels.

    psf_a, psf_b, psf_pa : float
        The shape parameters for the point spread function
        (degrees).

    flags : int
        Flags. See :module:`AegeanTools.flags`.

    uuid : str
        Unique ID for this source. This is random and not dependent on the source properties.

    See Also
    --------
    :module:`AegeanTools.flags`

    """
    #header for the output
    header = "#isl,src   bkg       rms         RA           DEC         RA         err         DEC        err         Peak      err     S_int     err        a    err    b    err     pa   err    flags\n" + \
             "#         Jy/beam   Jy/beam                               deg        deg         deg        deg       Jy/beam   Jy/beam    Jy       Jy         ''    ''    ''    ''    deg   deg   ZWNCPES\n" + \
             "#============================================================================================================================================================================================"

    #formatting strings for making nice output
    formatter = "({0.island:04d},{0.source:02d}) {0.background: 8.6f} {0.local_rms: 8.6f} " + \
                "{0.ra_str:12s} {0.dec_str:12s} {0.ra:11.7f} {0.err_ra: 9.7f} {0.dec:11.7f} {0.err_dec: 9.7f} " + \
                "{0.peak_flux: 8.6f} {0.err_peak_flux: 8.6f} {0.int_flux: 8.6f} {0.err_int_flux: 8.6f} " + \
                "{0.a:5.2f} {0.err_a:5.2f} {0.b:5.2f} {0.err_b:5.2f} " + \
                "{0.pa:6.1f} {0.err_pa:5.1f}   {0.flags:07b}"
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
        self.err_ra = np.nan  # degrees
        #dec = None # degrees
        self.err_dec = np.nan
        #peak_flux = None # Jy/beam
        #err_peak_flux = None # Jy/beam
        self.int_flux = np.nan  #Jy
        self.err_int_flux = np.nan  #Jy
        #self.a = 0.0 # major axis (arcsecs)
        self.err_a = np.nan  # arcsecs
        #self.b = 0.0 # minor axis (arcsecs)
        self.err_b = np.nan  # arcsecs
        #self.pa = 0.0 # position angle (degrees - WHAT??)
        self.err_pa = np.nan  # degrees
        self.flags = 0x0
        self.residual_mean = np.nan
        self.residual_std = np.nan
        #
        self.psf_a = np.nan
        self.psf_b = np.nan
        self.psf_pa = np.nan

    def __str__(self):
        self._sanitise()
        return self.formatter.format(self)

    def __repr__(self):
        return "({0:d},{1:d})".format(self.island, self.source)

    def __eq__(self, other):
        if self.island != other.island:
            return False
        if not hasattr(other, 'source'):
            return False
        return self.source == other.source

    def __ne__(self, other):
        if self.island != other.island:
            return True
        if not hasattr(other, 'source'):
            return True
        return self.source != other.source

    def __lt__(self, other):
        if not hasattr(other, 'island'):
            return True
        # Islands are always less than components
        if not hasattr(other, 'source'):
            return True
        if self.island < other.island:
            return True
        if self.island == other.island:
            return self.source < other.source

    def __le__(self, other):
        if not hasattr(other, 'island'):
            return True
        # Islands are always less than components
        if not hasattr(other, 'source'):
            return True
        if self.island < other.island:
            return True
        if self.island == other.island:
            return self.source <= other.source

    def __gt__(self, other):
        if not hasattr(other, 'island'):
            return False
        # Islands are always less than components
        if not hasattr(other, 'source'):
            return False
        if self.island > other.island:
            return True
        if self.island == other.island:
            return self.source > other.source

    def __ge__(self, other):
        if not hasattr(other, 'island'):
            return False
        # Islands are always less than components
        if not hasattr(other, 'source'):
            return False
        if self.island > other.island:
            return True
        if self.island == other.island:
            return self.source >= other.source


class GlobalFittingData(object):
    """
    A class to hold the properties associated with an image.
    [ These were once in the global scope of a monolithic script, hence the name].
    (should be) Read-only once created.
    Used by island fitting subprocesses.

    Attributes
    ----------
    img : :class:`AegeanTools.fits_image.FitsImage`
        Image that is being analysed, aka the input image.

    dcurve : 2d-array
        Image of +1,0,-1 representing the curvature of the input image.

    rmsimg, bkgimg : 2d-array
        The noise and background of the input image.

    hdu_header : HDUHeader
        FITS header for the input image.

    beam : :class:`AegeanTools.wcs_helpers.Beam`
        The synthesized beam of the input image.

    data_pix : 2d-array
        A link to the data array that is contained within the `img`.

    dtype : {np.float32, np.float64}
        The data type for the input image. Will be enforced upon writing.

    region : :class:`AegeanTools.regions.Region`
        The region that will be used to limit the source finding of Aegean.

    wcshelper : :class:`AegeanTools.wcs_helpers.WCSHelper`
        A helper object for WCS operations, created from `hdu_header`.

    blank : bool
        If true, then the input image will be blanked at the location of each of
        the measured islands.

    """

    def __init__(self):
        self.img = None
        self.dcurve = None
        self.rmsimg = None
        self.bkgimg = None
        self.hdu_header = None
        self.beam = None
        self.data_pix = None
        self.dtype = None
        self.region = None
        self.wcshelper = None
        self.psfhelper = None
        self.blank = False
        return


class PixelIsland(object):
    """
    An island of pixels within an image or cube

    Attributes
    ----------
    dim : int
        The number of dimensions of this island. dim >=2, default is 2 (ra/dec).

    bounding_box : [(min, max), (min, max), ...]
        A bounding box for this island. len(bounding_box)==dim.

    mask : np.array(dtype=bool)
        A mask that represents the island within the bounding box.
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.bounding_box = np.zeros((self.dim,2), dtype=np.int32)
        self.mask = None
        self.partial = False
        return

    def set_mask(self, data):
        """

        Parameters
        ----------
        data : np.array
        """
        if len(data.shape) != self.dim:
            raise AssertionError("mask shape {0} is of the wrong dimension. Expecting {1}".format(data.shape, self.dim))
        self.mask = data
        return

    def calc_bounding_box(self, data, offsets):
        """
        Compute the bounding box for a data cube of dimension dim.
        The bounding box will be the smallest nd-cube that bounds the non-zero entries of the cube.
        Parameters
        ----------
        data : np.ndarray
            Data array with dimension equal to self.dim

        offsets : [xmin, ymin, ...]
            The offset between the image zero index and the zero index of data. len(offsets)==dim
        """
        if len(offsets)!=self.dim:
            raise AssertionError("{0} offsets were passed but {1} are required".format(len(offsets),self.dim))
        # TODO: Figure out 3d boxes
        # set the bounding box one dimension at a time
        ndrow = np.any(data, axis=0)
        rmin, rmax = np.where(ndrow)[0][[0, -1]]
        self.bounding_box[1][0] = offsets[1] + rmin
        self.bounding_box[1][1] = offsets[1] + rmax + 1
        ndcol = np.any(data, axis=1)
        cmin, cmax = np.where(ndcol)[0][[0, -1]]
        self.bounding_box[0][0] = offsets[0] + cmin
        self.bounding_box[0][1] = offsets[0] + cmax + 1
        self.set_mask(data[rmin:rmax+1, cmin:cmax+1])
        return


class IslandFittingData(object):
    """
    All the data required to fit a single island.
    Instances are pickled and passed to the fitting subprocesses

    Attributes
    ----------
    isle_num : int
        island number

    i : 2d-array
        a 2D numpy array of pixel values

    scalars : (innerclip, outerclip, max_summits)
        Inner and outer clipping limits (sigma), and the maximum number of components that should be fit.

    offsets : (xmin, xmax, ymin, ymax)
        The offset between the boundaries of the island i, within the
        larger image.

    doislandflux : boolean
        If true then also measure properties of the island.
    """

    def __init__(self, isle_num=0, i=None, scalars=None, offsets=(0,0,1,1), doislandflux=False):
        self.isle_num = isle_num
        self.i = i
        self.scalars = scalars
        self.offsets = offsets
        self.doislandflux = doislandflux


class DummyLM(object):
    """
    A dummy copy of the lmfit results, for use when no fitting was done.

    Attributes
    ----------
    residual : [np.nan, np.nan]
        The residual background and rms.

    success: bool
        False - the fitting has failed.
    """

    def __init__(self):
        self.residual = [np.nan, np.nan]
        self.success = False


def classify_catalog(catalog):
    """
    Look at a list of sources and split them according to their class.

    Parameters
    ----------
    catalog : iterable
        A list or iterable object of {SimpleSource, IslandSource, ComponentSource} objects, possibly mixed.
        Any other objects will be silently ignored.

    Returns
    -------
    components : list
        List of sources of type ComponentSource

    islands : list
        List of sources of type IslandSource

    simples : list
        List of source of type SimpleSource
    """
    components = []
    islands = []
    simples = []
    for source in catalog:
        if isinstance(source, ComponentSource):
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

    Parameters
    ----------
    catalog : iterable
        A list or iterable of :class:`AegeanTools.models.ComponentSource` objects.

    Yields
    ------
    group : list
        A list of all sources within an island, one island at a time.

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
