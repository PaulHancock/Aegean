#! /usr/bin/env python
"""
Module for reading at writing catalogs
"""

from __future__ import print_function

__author__ = "Paul Hancock"
__version__ = "1.0"
__date__ = "2016-07-26"

# Standard imports
import os
import numpy as np
import re
import six
from time import gmtime, strftime

# Other AegeanTools
from .models import ComponentSource, classify_catalog

# input/output table formats
from astropy.table.table import Table
from astropy.io import ascii
from astropy.io import fits
from astropy.io.votable import from_table, parse_single_table
from astropy.io.votable import writeto as writetoVO

# try:
#     import h5py
#
#     hdf5_supported = True
# except ImportError:
#     hdf5_supported = False

import sqlite3

# join the Aegean logger
import logging

log = logging.getLogger('Aegean')


# writing table formats
def check_table_formats(files):
    """
    Determine whether a list of files are of a recognizable output type.

    Parameters
    ----------
    files : str
        A list of file names

    Returns
    -------
    result : bool
        True if *all* the file names are supported
    """
    cont = True
    formats = get_table_formats()
    for t in files.split(','):
        _, ext = os.path.splitext(t)
        ext = ext[1:].lower()
        if ext not in formats:
            cont = False
            log.warning("Format not supported for {0} ({1})".format(t, ext))
    if not cont:
        log.error("Invalid table format specified.")
    return cont


def show_formats():
    """
    Print a list of all the file formats that are supported for writing.
    The file formats are determined by their extensions.

    Returns
    -------
    None
    """
    fmts = {
        "ann": "Kvis annotation",
        "reg": "DS9 regions file",
        "fits": "FITS Binary Table",
        "csv": "Comma separated values",
        "tab": "tabe separated values",
        "tex": "LaTeX table format",
        "html": "HTML table",
        "vot": "VO-Table",
        "xml": "VO-Table",
        "db": "Sqlite3 database",
        "sqlite": "Sqlite3 database"}
    supported = get_table_formats()
    print("Extension |     Description       | Supported?")
    for k in sorted(fmts.keys()):
        print("{0:10s} {1:24s} {2}".format(k, fmts[k], k in supported))
    return


def get_table_formats():
    """
    Create a list of file extensions that are supported for writing.

    Returns
    -------
    fmts : list
        A list of file name extensions that are supported.
    """
    fmts = ['reg', 'fits']
    fmts.extend(['vo', 'vot', 'xml'])
    fmts.extend(['csv', 'tab', 'tex', 'html'])
    # if hdf5_supported:
    #     fmts.append('hdf5')
    # else:
    #     log.info("HDF5 is not supported by your environment")
    # assume this is always possible -> though it may not be on some systems
    fmts.extend(['db', 'sqlite'])
    return fmts


def update_meta_data(meta=None):
    """
    Modify the metadata dictionary.
    DATE, PROGRAM, and PROGVER are added/modified.

    Parameters
    ----------
    meta : dict
        The dictionary to be modified, default = None (empty)

    Returns
    -------
        An updated dictionary.
    """
    if meta is None:
        meta = {}
    if 'DATE' not in meta:
        meta['DATE'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    if 'PROGRAM' not in meta:
        meta['PROGRAM'] = "AegeanTools.catalogs"
        meta['PROGVER'] = "{0}-({1})".format(__version__, __date__)
    return meta


def save_catalog(filename, catalog, meta=None, prefix=None):
    """
    Save a catalogue of sources using filename as a model. Meta data can be written to some file types
    (fits, votable).

    Each type of source will be in a separate file:

    - base_comp.ext :class:`AegeanTools.models.ComponentSource`
    - base_isle.ext :class:`AegeanTools.models.IslandSource`
    - base_simp.ext :class:`AegeanTools.models.SimpleSource`


    Where filename = `base.ext`

    Parameters
    ----------
    filename : str
        Name of file to write, format is determined by extension.

    catalog : list
        A list of sources to write. Sources must be of type :class:`AegeanTools.models.ComponentSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    prefix : str
        Prepend each column name with "prefix_". Default is to prepend nothing.

    meta : dict
        Meta data to be written to the output file. Support for metadata depends on file type.

    Returns
    -------
    None
    """
    ascii_table_formats = {'csv': 'csv', 'tab': 'tab', 'tex': 'latex', 'html': 'html'}
    # .ann and .reg are handled by me
    meta = update_meta_data(meta)
    extension = os.path.splitext(filename)[1][1:].lower()
    if extension in ['ann', 'reg']:
        writeAnn(filename, catalog, extension)
    elif extension in ['db', 'sqlite']:
        writeDB(filename, catalog, meta)
    elif extension in ['hdf5', 'fits', 'vo', 'vot', 'xml']:
        write_catalog(filename, catalog, extension, meta, prefix=prefix)
    elif extension in ascii_table_formats.keys():
        write_catalog(filename, catalog, fmt=ascii_table_formats[extension], meta=meta, prefix=prefix)
    else:
        log.warning("extension not recognised {0}".format(extension))
        log.warning("You get tab format")
        write_catalog(filename, catalog, fmt='tab', prefix=prefix)
    return


def load_catalog(filename):
    """
    Load a catalogue and extract the source positions (only)

    Parameters
    ----------
    filename : str
        Filename to read. Supported types are csv, tab, tex, vo, vot, and xml.

    Returns
    -------
    catalogue : list
        A list of [ (ra, dec), ...]

    """
    supported = get_table_formats()

    fmt = os.path.splitext(filename)[-1][1:].lower()  # extension sans '.'

    if fmt in ['csv', 'tab', 'tex'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = ascii.read(filename)
        catalog = list(zip(t.columns['ra'], t.columns['dec']))

    elif fmt in ['vo', 'vot', 'xml'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = parse_single_table(filename)
        catalog = list(zip(t.array['ra'].tolist(), t.array['dec'].tolist()))

    else:
        log.info("Assuming ascii format, reading first two columns")
        lines = [a.strip().split() for a in open(filename, 'r').readlines() if not a.startswith('#')]
        try:
            catalog = [(float(a[0]), float(a[1])) for a in lines]
        except:
            log.error("Expecting two columns of floats but failed to parse")
            log.error("Catalog file {0} not loaded".format(filename))
            raise Exception("Could not determine file format")

    return catalog


def load_table(filename):
    """
    Load a table from a given file.

    Supports csv, tab, tex, vo, vot, xml, fits, and hdf5.

    Parameters
    ----------
    filename : str
        File to read

    Returns
    -------
    table : Table
        Table of data.
    """
    supported = get_table_formats()

    fmt = os.path.splitext(filename)[-1][1:].lower()  # extension sans '.'

    if fmt in ['csv', 'tab', 'tex'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = ascii.read(filename)
    elif fmt in ['vo', 'vot', 'xml', 'fits', 'hdf5'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = Table.read(filename)
    else:
        log.error("Table format not recognized or supported")
        log.error("{0} [{1}]".format(filename, fmt))
        raise Exception("Table format not recognized or supported")
    return t


def write_table(table, filename):
    """
    Write a table to a file.

    Parameters
    ----------
    table : Table
        Table to be written

    filename : str
        Destination for saving table.

    Returns
    -------
    None
    """
    try:
        if os.path.exists(filename):
            os.remove(filename)
        table.write(filename)
        log.info("Wrote {0}".format(filename))
    except Exception as e:
        if "Format could not be identified" not in e.message:
            raise e
        else:
            fmt = os.path.splitext(filename)[-1][1:].lower()  # extension sans '.'
            raise Exception("Cannot auto-determine format for {0}".format(fmt))
    return


def table_to_source_list(table, src_type=ComponentSource):
    """
    Convert a table of data into a list of sources.

    A single table must have consistent source types given by src_type. src_type should be one of
    :class:`AegeanTools.models.ComponentSource`, :class:`AegeanTools.models.SimpleSource`,
    or :class:`AegeanTools.models.IslandSource`.


    Parameters
    ----------
    table : Table
        Table of sources

    src_type : class
        Sources must be of type :class:`AegeanTools.models.ComponentSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    Returns
    -------
    sources : list
        A list of objects of the given type.
    """
    source_list = []
    if table is None:
        return source_list

    for row in table:
        # Initialise our object
        src = src_type()
        # look for the columns required by our source object
        for param in src_type.names:
            if param in table.colnames:
                # copy the value to our object
                val = row[param]
                # hack around float32's broken-ness
                if isinstance(val, np.float32):
                    val = np.float64(val)
                setattr(src, param, val)
        # save this object to our list of sources
        source_list.append(src)
    return source_list


def write_catalog(filename, catalog, fmt=None, meta=None, prefix=None):
    """
    Write a catalog (list of sources) to a file with format determined by extension.

    Sources must be of type :class:`AegeanTools.models.ComponentSource`,
    :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    Parameters
    ----------
    filename : str
        Base name for file to write. `_simp`, `_comp`, or `_isle` will be added to differentiate
        the different types of sources that are being written.

    catalog : list
        A list of source objects. Sources must be of type :class:`AegeanTools.models.ComponentSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    fmt : str
        The file format extension.

    prefix : str
        Prepend each column name with "prefix_". Default is to prepend nothing.

    meta : dict
        A dictionary to be used as metadata for some file types (fits, VOTable).

    Returns
    -------
    None
    """
    if meta is None:
        meta = {}

    if prefix is None:
        pre=''
    else:
        pre = prefix + '_'

    def writer(filename, catalog, fmt=None):
        """
        construct a dict of the data
        this method preserves the data types in the VOTable
        """
        tab_dict = {}
        name_list = []
        for name in catalog[0].names:
            col_name = name
            if catalog[0].galactic:
                if name.startswith('ra'):
                    col_name = 'lon'+name[2:]
                elif name.endswith('ra'):
                    col_name = name[:-2] + 'lon'
                elif name.startswith('dec'):
                    col_name = 'lat'+name[3:]
                elif name.endswith('dec'):
                    col_name = name[:-3] + 'lat'
            col_name = pre + col_name
            tab_dict[col_name] = [getattr(c, name, None) for c in catalog]
            name_list.append(col_name)
        t = Table(tab_dict, meta=meta)
        # re-order the columns
        t = t[[n for n in name_list]]

        if fmt is not None:
            if fmt in ["vot", "vo", "xml"]:
                vot = from_table(t)
                # description of this votable
                vot.description = repr(meta)
                writetoVO(vot, filename)
            elif fmt in ['hdf5']:
                t.write(filename, path='data', overwrite=True)
            elif fmt in ['fits']:
                writeFITSTable(filename, t)
            else:
                ascii.write(t, filename, fmt, overwrite=True)
        else:
            ascii.write(t, filename, overwrite=True)
        return

    # sort the sources into types and then write them out individually
    components, islands, simples = classify_catalog(catalog)

    if len(components) > 0:
        new_name = "{1}{0}{2}".format('_comp', *os.path.splitext(filename))
        writer(new_name, components, fmt)
        log.info("wrote {0}".format(new_name))
    if len(islands) > 0:
        new_name = "{1}{0}{2}".format('_isle', *os.path.splitext(filename))
        writer(new_name, islands, fmt)
        log.info("wrote {0}".format(new_name))
    if len(simples) > 0:
        new_name = "{1}{0}{2}".format('_simp', *os.path.splitext(filename))
        writer(new_name, simples, fmt)
        log.info("wrote {0}".format(new_name))
    return


def writeFITSTable(filename, table):
    """
    Convert a table into a FITSTable and then write to disk.

    Parameters
    ----------
    filename : str
        Filename to write.

    table : Table
        Table to write.

    Returns
    -------
    None

    Notes
    -----
    Due to a bug in numpy, `int32` and `float32` are converted to `int64` and `float64` before writing.
    """
    def FITSTableType(val):
        """
        Return the FITSTable type corresponding to each named parameter in obj
        """
        if isinstance(val, bool):
            types = "L"
        elif isinstance(val, (int, np.int64, np.int32)):
            types = "J"
        elif isinstance(val, (float, np.float64, np.float32)):
            types = "E"
        elif isinstance(val, six.string_types):
            types = "{0}A".format(len(val))
        else:
            log.warning("Column {0} is of unknown type {1}".format(val, type(val)))
            log.warning("Using 5A")
            types = "5A"
        return types

    cols = []
    for name in table.colnames:
        # Cause error columns to always be floats even when they are set to -1
        if name.startswith('err_'):
            fmt = 'E'
        elif name == 'uuid':
            fmt = '{0}A'.format(max(len(val) for val in table[name]))
        else:
            fmt = FITSTableType(table[name][0])
        cols.append(fits.Column(name=name, format=fmt, array=table[name]))
    cols = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    for k in table.meta:
        tbhdu.header['HISTORY'] = ':'.join((k, table.meta[k]))
    tbhdu.writeto(filename, overwrite=True)


def writeIslandContours(filename, catalog, fmt='reg'):
    """
    Write an output file in ds9 .reg format that outlines the boundaries of each island.

    Parameters
    ----------
    filename : str
        Filename to write.

    catalog : list
        List of sources. Only those of type :class:`AegeanTools.models.IslandSource` will have contours drawn.

    fmt : str
        Output format type. Currently only 'reg' is supported (default)

    Returns
    -------
    None

    See Also
    --------
    :func:`AegeanTools.catalogs.writeIslandBoxes`
    """
    if fmt != 'reg':
        log.warning("Format {0} not yet supported".format(fmt))
        log.warning("not writing anything")
        return

    out = open(filename, 'w')
    print("#Aegean island contours", file=out)
    print("#AegeanTools.catalogs version {0}-({1})".format(__version__, __date__), file=out)
    line_fmt = 'image;line({0},{1},{2},{3})'
    text_fmt = 'fk5; text({0},{1}) # text={{{2}}}'
    mas_fmt = 'image; line({1},{0},{3},{2}) #color = yellow'
    x_fmt = 'image; point({1},{0}) # point=x'
    for c in catalog:
        contour = c.contour
        if len(contour) > 1:
            for p1, p2 in zip(contour[:-1], contour[1:]):
                print(line_fmt.format(p1[1] + 0.5, p1[0] + 0.5, p2[1] + 0.5, p2[0] + 0.5), file=out)
            print(line_fmt.format(contour[-1][1] + 0.5, contour[-1][0] + 0.5, contour[0][1] + 0.5,
                                          contour[0][0] + 0.5), file=out)
        # comment out lines that have invalid ra/dec (WCS problems)
        if np.nan in [c.ra, c.dec]:
            print('#', end=' ', file=out)
        # some islands may not have anchors because they don't have any contours
        if len(c.max_angular_size_anchors) == 4:
            print(text_fmt.format(c.ra, c.dec, c.island), file=out)
            print(mas_fmt.format(*[a + 0.5 for a in c.max_angular_size_anchors]), file=out)
        for p1, p2 in c.pix_mask:
            # DS9 uses 1-based instead of 0-based indexing
            print(x_fmt.format(p1 + 1, p2 + 1), file=out)
    out.close()
    return


def writeIslandBoxes(filename, catalog, fmt):
    """
    Write an output file in ds9 .reg, or kvis .ann format that contains bounding boxes for all the islands.

    Parameters
    ----------
    filename : str
        Filename to write.

    catalog : list
        List of sources. Only those of type :class:`AegeanTools.models.IslandSource` will have contours drawn.

    fmt : str
        Output format type. Currently only 'reg' and 'ann' are supported. Default = 'reg'.

    Returns
    -------
    None

    See Also
    --------
    :func:`AegeanTools.catalogs.writeIslandContours`
    """
    if fmt not in ['reg', 'ann']:
        log.warning("Format not supported for island boxes{0}".format(fmt))
        return  # fmt not supported

    out = open(filename, 'w')
    print("#Aegean Islands", file=out)
    print("#Aegean version {0}-({1})".format(__version__, __date__), file=out)

    if fmt == 'reg':
        print("IMAGE", file=out)
        box_fmt = 'box({0},{1},{2},{3}) #{4}'
    else:
        print("COORD P", file=out)
        box_fmt = 'box P {0} {1} {2} {3} #{4}'

    for c in catalog:
        # x/y swap for pyfits/numpy translation
        ymin, ymax, xmin, xmax = c.extent
        # +1 for array/image offset
        xcen = (xmin + xmax) / 2.0 + 1
        # + 0.5 in each direction to make lines run 'between' DS9 pixels
        xwidth = xmax - xmin + 1
        ycen = (ymin + ymax) / 2.0 + 1
        ywidth = ymax - ymin + 1
        print(box_fmt.format(xcen, ycen, xwidth, ywidth, c.island), file=out)
    out.close()
    return


def writeAnn(filename, catalog, fmt):
    """
    Write an annotation file that can be read by Kvis (.ann) or DS9 (.reg).
    Uses ra/dec from catalog.
    Draws ellipses if bmaj/bmin/pa are in catalog. Draws 30" circles otherwise.

    Only :class:`AegeanTools.models.ComponentSource` will appear in the annotation file
    unless there are none, in which case :class:`AegeanTools.models.SimpleSource` (if present)
    will be written. If any :class:`AegeanTools.models.IslandSource` objects are present then
    an island contours file will be written.

    Parameters
    ----------
    filename : str
        Output filename base.

    catalog : list
        List of sources.

    fmt : ['ann', 'reg']
        Output file type.

    Returns
    -------
    None

    See Also
    --------
    AegeanTools.catalogs.writeIslandContours
    """
    if fmt not in ['reg', 'ann']:
        log.warning("Format not supported for island boxes{0}".format(fmt))
        return  # fmt not supported

    components, islands, simples = classify_catalog(catalog)
    if len(components) > 0:
        cat = sorted(components)
        suffix = "comp"
    elif len(simples) > 0:
        cat = simples
        suffix = "simp"
    else:
        cat = []

    if len(cat) > 0:
        ras = [a.ra for a in cat]
        decs = [a.dec for a in cat]
        if not hasattr(cat[0], 'a'):  # a being the variable that I used for bmaj.
            bmajs = [30 / 3600.0 for a in cat]
            bmins = bmajs
            pas = [0 for a in cat]
        else:
            bmajs = [a.a / 3600.0 for a in cat]
            bmins = [a.b / 3600.0 for a in cat]
            pas = [a.pa for a in cat]

        names = [a.__repr__() for a in cat]
        if fmt == 'ann':
            new_file = re.sub('.ann$', '_{0}.ann'.format(suffix), filename)
            out = open(new_file, 'w')
            print("#Aegean version {0}-({1})".format(__version__, __date__), file=out)
            print('PA SKY', file=out)
            print('FONT hershey12', file=out)
            print('COORD W', file=out)
            formatter = "ELLIPSE W {0} {1} {2} {3} {4:+07.3f} #{5}\nTEXT W {0} {1} {5}"
        else:  # reg
            new_file = re.sub('.reg$', '_{0}.reg'.format(suffix), filename)
            out = open(new_file, 'w')
            print("#Aegean version {0}-({1})".format(__version__, __date__), file=out)
            print("fk5", file=out)
            formatter = 'ellipse {0} {1} {2:.9f}d {3:.9f}d {4:+07.3f}d # text="{5}"'
            # DS9 has some strange ideas about position angle
            pas = [a - 90 for a in pas]

        for ra, dec, bmaj, bmin, pa, name in zip(ras, decs, bmajs, bmins, pas, names):
            # comment out lines that have invalid or stupid entries
            if np.nan in [ra, dec, bmaj, bmin, pa] or bmaj >= 180:
                print('#', end=' ', file=out)
            print(formatter.format(ra, dec, bmaj, bmin, pa, name), file=out)
        out.close()
        log.info("wrote {0}".format(new_file))
    if len(islands) > 0:
        if fmt == 'reg':
            new_file = re.sub('.reg$', '_isle.reg', filename)
        elif fmt == 'ann':
            log.warning('kvis islands are currently not working')
            return
        else:
            log.warning('format {0} not supported for island annotations'.format(fmt))
            return
        writeIslandContours(new_file, islands, fmt)
        log.info("wrote {0}".format(new_file))

    return


def nulls(x):
    """
    Convert values of -1 into None.

    Parameters
    ----------
    x : float or int
        Value to convert

    Returns
    -------
    val : [x, None]
    """
    if x == -1:
        return None
    else:
        return x


def writeDB(filename, catalog, meta=None):
    """
    Output an sqlite3 database containing one table for each source type

    Parameters
    ----------
    filename : str
        Output filename

    catalog : list
        List of sources of type :class:`AegeanTools.models.ComponentSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    meta : dict
        Meta data to be written to table `meta`

    Returns
    -------
    None
    """

    def sqlTypes(obj, names):
        """
        Return the sql type corresponding to each named parameter in obj
        """
        types = []
        for n in names:
            val = getattr(obj, n)
            if isinstance(val, bool):
                types.append("BOOL")
            elif isinstance(val, (int, np.int64, np.int32)):
                types.append("INT")
            elif isinstance(val, (float, np.float64, np.float32)):  # float32 is bugged and claims not to be a float
                types.append("FLOAT")
            elif isinstance(val, six.string_types):
                types.append("VARCHAR")
            else:
                log.warning("Column {0} is of unknown type {1}".format(n, type(n)))
                log.warning("Using VARCHAR")
                types.append("VARCHAR")
        return types

    if os.path.exists(filename):
        log.warning("overwriting {0}".format(filename))
        os.remove(filename)
    conn = sqlite3.connect(filename)
    db = conn.cursor()
    # determine the column names by inspecting the catalog class
    for t, tn in zip(classify_catalog(catalog), ["components", "islands", "simples"]):
        if len(t) < 1:
            continue  #don't write empty tables
        col_names = t[0].names
        col_types = sqlTypes(t[0], col_names)
        stmnt = ','.join(["{0} {1}".format(a, b) for a, b in zip(col_names, col_types)])
        db.execute('CREATE TABLE {0} ({1})'.format(tn, stmnt))
        stmnt = 'INSERT INTO {0} ({1}) VALUES ({2})'.format(tn, ','.join(col_names), ','.join(['?' for i in col_names]))
        # expend the iterators that are created by python 3+
        data = list(map(nulls, list(r.as_list() for r in t)))
        db.executemany(stmnt, data)
        log.info("Created table {0}".format(tn))
    # metadata add some meta data
    db.execute("CREATE TABLE meta (key VARCHAR, val VARCHAR)")
    for k in meta:
        db.execute("INSERT INTO meta (key, val) VALUES (?,?)", (k, meta[k]))
    conn.commit()
    log.info(db.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall())
    conn.close()
    log.info("Wrote file {0}".format(filename))
    return

