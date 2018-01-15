#! python
from __future__ import print_function

__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import catalogs as cat
from AegeanTools.models import OutputSource, IslandSource, SimpleSource
from AegeanTools.msq2 import MarchingSquares
import numpy as np
from numpy.testing import assert_raises
import os

import logging
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logging.INFO)


def test_check_table_formats():
    files = ','.join(['a.csv', 'a.fits', 'a.vot', 'a.hdf5',  'a.ann', 'a.docx', 'a'])

    if cat.check_table_formats(files):
        raise AssertionError()
    if not cat.check_table_formats('files.fits'):
        raise AssertionError()


def test_show_formats():
    cat.show_formats()


def test_get_table_formats():
    formats = cat.get_table_formats()
    for f in formats:
        name = 'a.'+f
        if not cat.check_table_formats(name):
            raise AssertionError()


def test_update_meta_data():
    meta = None
    meta = cat.update_meta_data(meta)
    if 'PROGRAM' not in meta:
        raise AssertionError()

    meta = {'DATE': 1}
    meta = cat.update_meta_data(meta)
    if not meta['DATE'] == 1:
        raise AssertionError()


def test_load_save_catalog():
    catalog = [OutputSource()]
    for ext in ['csv', 'vot']:
        fout = 'a.'+ext
        cat.save_catalog(fout, catalog, meta=None)
        fout = 'a_comp.'+ext
        if not os.path.exists(fout):
            raise AssertionError()

        catin = cat.load_catalog(fout)
        if not len(catin) == len(catalog):
            raise AssertionError()

        os.remove(fout)

    for ext in ['reg', 'ann', 'bla']:
        fout = 'a.'+ext
        cat.save_catalog(fout, catalog, meta=None)
        fout = 'a_comp.'+ext
        if not os.path.exists(fout):
            raise AssertionError()

        os.remove(fout)

    fout = 'a.db'
    cat.save_catalog(fout, catalog, meta=None)
    if not os.path.exists(fout):
        raise AssertionError()

    os.remove(fout)

    badfile = open("file.fox", 'w')
    print("ff 012.d", file=badfile)
    badfile.close()
    assert_raises(Exception, cat.load_catalog, 'file.fox')
    badfile = open("file.fox", 'w')
    print('1 1', file=badfile)
    badfile.close()
    catin = cat.load_catalog('file.fox')
    print(catin)
    if not len(catin) == 1:
        raise AssertionError()

    os.remove('file.fox')


def test_load_table_write_table():
    catalog = [OutputSource()]
    for fmt in ['csv', 'vo']:
        fout = 'a.'+fmt
        cat.save_catalog(fout, catalog, meta=None)
        fout = 'a_comp.'+fmt
        tab = cat.load_table(fout)
        if not len(tab) == len(catalog):
            raise AssertionError()

        os.remove(fout)

    cat.save_catalog('a.csv', catalog, meta=None)
    tab = cat.load_table('a_comp.csv')
    cat.write_table(tab, 'a.csv')
    if not os.path.exists('a.csv'):
        raise AssertionError()

    os.remove('a.csv')

    assert_raises(Exception, cat.write_table, tab, 'bla.fox')
    assert_raises(Exception, cat.load_table, 'file.fox')


def test_write_comp_isl_simp():
    catalog = [OutputSource(), IslandSource(), SimpleSource()]
    catalog[0].galactic = True
    out = 'a.csv'
    cat.write_catalog(out, catalog)
    if not os.path.exists('a_isle.csv'):
        raise AssertionError()

    os.remove('a_isle.csv')
    if not os.path.exists('a_comp.csv'):
        raise AssertionError()

    os.remove('a_comp.csv')
    if not os.path.exists('a_simp.csv'):
        raise AssertionError()

    os.remove('a_simp.csv')


def dont_test_load_save_fits_tables():
    # The version of astropy on travis breaks on this!
    # probably a bug that will be fixed by astropy later.
    catalog = [OutputSource()]
    cat.save_catalog('a.fits', catalog, meta=None)
    if not os.path.exists('a_comp.fits'):
        raise AssertionError()

    os.remove('a_comp.fits')
    # Somehow this doesn't work for my simple test cases
    # catin = cat.load_table('a_comp.fits')
    # assert len(catin) == 2


def test_write_contours_boxes():
    data = np.zeros((5, 5))
    data[1:4, 2] = 1.
    data[2, 1:4] = 1.
    ms = MarchingSquares(data)
    src = IslandSource()
    src.contour = ms.perimeter
    src.extent = [1, 4, 1, 4]
    catalog = [src]
    cat.writeIslandContours('out.reg', catalog, fmt='reg')
    if not os.path.exists('out.reg'):
        raise AssertionError()

    os.remove('out.reg')
    # shouldn't write anything
    cat.writeIslandContours('out.ann', catalog, fmt='ann')
    if os.path.exists('out.ann'):
        raise AssertionError()

    cat.writeIslandBoxes('out.reg', catalog, fmt='reg')
    if not os.path.exists('out.reg'):
        raise AssertionError()

    os.remove('out.reg')
    cat.writeIslandBoxes('out.ann', catalog, fmt='ann')
    if not os.path.exists('out.ann'):
        raise AssertionError()

    os.remove('out.ann')
    # shouldn't write anything
    cat.writeIslandBoxes('out.ot', catalog, fmt='ot')
    if os.path.exists('out.ot'):
        raise AssertionError()


def test_write_ann():
    # write regular and simple sources for .ann files
    cat.writeAnn('out.ann', [OutputSource()], fmt='ann')
    if not os.path.exists('out_comp.ann'):
        raise AssertionError()

    os.remove('out_comp.ann')
    cat.writeAnn('out.ann', [SimpleSource()], fmt='ann')
    if not os.path.exists('out_simp.ann'):
        raise AssertionError()

    os.remove('out_simp.ann')
    # same but for .reg files
    cat.writeAnn('out.reg', [OutputSource()], fmt='reg')
    if not os.path.exists('out_comp.reg'):
        raise AssertionError()

    os.remove('out_comp.reg')
    cat.writeAnn('out.reg', [SimpleSource()], fmt='reg')
    if not os.path.exists('out_simp.reg'):
        raise AssertionError()

    os.remove('out_simp.reg')


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()