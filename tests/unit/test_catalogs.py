#! /usr/bin/env python
"""
Test catalogues.py
"""

import logging
import os

import numpy as np
from AegeanTools import catalogs as cat
from AegeanTools.models import ComponentSource, IslandSource, SimpleSource
from AegeanTools.msq2 import MarchingSquares
from astropy import table
from numpy.testing import assert_raises

__author__ = 'Paul Hancock'

logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")
log.setLevel(logger.info)


def test_nulls():
    if cat.nulls(-1) is not None:
        raise AssertionError("nulls(-1) is broken")
    if cat.nulls(0) is None:
        raise AssertionError("nulls(0) is broken")


def test_check_table_formats():
    """Test check_table_formats"""
    files = ','.join(['a.csv', 'a.fits', 'a.vot',
                     'a.hdf5',  'a.ann', 'a.docx', 'a'])

    if cat.check_table_formats(files):
        raise AssertionError()
    if not cat.check_table_formats('files.fits'):
        raise AssertionError()


def test_show_formats():
    """Test that show_formats doesn't crash"""
    cat.show_formats()


def test_get_table_formats():
    """Test get_table_formats"""
    formats = cat.get_table_formats()
    for f in formats:
        name = 'a.'+f
        if not cat.check_table_formats(name):
            raise AssertionError()


def test_update_meta_data():
    """Test that update_meta_data adds the desired keys"""
    meta = None
    meta = cat.update_meta_data(meta)
    if 'PROGRAM' not in meta:
        raise AssertionError()

    meta = {'DATE': 1}
    meta = cat.update_meta_data(meta)
    if not meta['DATE'] == 1:
        raise AssertionError()


def test_load_save_catalog():
    """Test that we can load and save various file formats"""
    catalog = [ComponentSource()]
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

    # test the prefix is being written.
    fout = 'a.csv'
    cat.save_catalog(fout, catalog, meta=None, prefix='test')
    fout = 'a_comp.csv'
    if not os.path.exists(fout):
        raise AssertionError()
    if 'test_ra' not in open(fout).readlines()[0]:
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

    # again so that we trigger an overwrite
    cat.save_catalog(fout, catalog, meta=None)
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
    """Test that we can write and load tables with various file formats"""
    catalog = [ComponentSource()]
    for fmt in ['csv', 'vo']:
        fout = 'a.'+fmt
        cat.save_catalog(fout, catalog, meta=None)
        fout = 'a_comp.'+fmt
        tab = cat.load_table(fout)
        if not len(tab) == len(catalog):
            raise AssertionError()

    # by keeping this out of the loop, we make use of the internal remove function
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
    """Test the writing of components/islands/simples to csv format files"""
    catalog = [ComponentSource(), IslandSource(), SimpleSource()]
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
    """Test that we can load and save fits tables"""
    # The version of astropy on travis breaks on this!
    # probably a bug that will be fixed by astropy later.
    catalog = [ComponentSource()]
    cat.save_catalog('a.fits', catalog, meta=None)
    if not os.path.exists('a_comp.fits'):
        raise AssertionError()

    os.remove('a_comp.fits')
    # Somehow this doesn't work for my simple test cases
    # catin = cat.load_table('a_comp.fits')
    # assert len(catin) == 2


def test_write_fits_table_variable_uuid_lengths():
    """Test that the length of the UUID column is appropriate"""
    catalog = []
    for l in range(10):
        c = ComponentSource()
        c.ra_str = c.dec_str = "hello!"
        c.uuid = 'source-{0:d}'.format(2**l)
        catalog.append(c)
    cat.save_catalog('a.fits', catalog, meta={'Purpose': 'Testing'})
    if not os.path.exists('a_comp.fits'):
        raise AssertionError()

    rcat = cat.load_table('a_comp.fits')
    for src1, src2 in zip(rcat, catalog):
        if len(src1['uuid']) != len(src2.uuid):
            print("len mismatch for source {0}".format(src1))
            print("uuid should be len={0}".format(len(src2.uuid)))
            raise AssertionError("UUID col is of wrong length")
    os.remove('a_comp.fits')
    return


def test_write_contours_boxes():
    """Test that we can write contour boxes for our island sources"""
    data = np.zeros((5, 5))
    data[1:4, 2] = 1.
    data[2, 1:4] = 1.
    ms = MarchingSquares(data)
    src = IslandSource()
    src.contour = ms.perimeter
    src.max_angular_size_anchors = [1, 2, 3, 4]
    src.pix_mask = [[0, 0], [1, 1]]
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
    """Test that write_ann *doesn't* do anything"""
    cat.writeAnn('out.ann', [], fmt='fail')
    if os.path.exists('out.ann'):
        raise AssertionError("Shoudn't have written anything")

    src = ComponentSource()
    # remove the parameter a to hit a checkpoint
    del src.a
    cat.writeAnn('out.reg', [src], fmt='reg')
    if not os.path.exists('out_comp.reg'):
        raise AssertionError()
    os.remove('out_comp.reg')

    # write regular and simple sources for .ann files
    cat.writeAnn('out.ann', [ComponentSource()], fmt='ann')
    if not os.path.exists('out_comp.ann'):
        raise AssertionError()

    os.remove('out_comp.ann')
    cat.writeAnn('out.ann', [SimpleSource()], fmt='ann')
    if not os.path.exists('out_simp.ann'):
        raise AssertionError()

    os.remove('out_simp.ann')
    # same but for .reg files
    cat.writeAnn('out.reg', [ComponentSource()], fmt='reg')
    if not os.path.exists('out_comp.reg'):
        raise AssertionError()

    os.remove('out_comp.reg')
    cat.writeAnn('out.reg', [SimpleSource()], fmt='reg')
    if not os.path.exists('out_simp.reg'):
        raise AssertionError()

    os.remove('out_simp.reg')
    cat.writeAnn('out.reg', [IslandSource()], fmt='reg')
    if not os.path.exists('out_isle.reg'):
        raise AssertionError()

    os.remove('out_isle.reg')
    cat.writeAnn('out.ann', [IslandSource()], fmt='ann')
    cat.writeAnn('out.reg', [IslandSource()], fmt='fail')


def test_table_to_source_list():
    """Test that we can convert empty tables to an empty source list"""
    slist = cat.table_to_source_list(None)
    if not (slist == []):
        raise AssertionError()


def test_writeFITSTable():
    """Test that we can write a fits table"""
    tab = table.Table.read('tests/test_files/1904_comp.fits')
    outfile = 'dlme.fits'
    tab.meta = {'test': 'test'}
    cat.writeFITSTable(outfile, tab)
    if not os.path.exists(outfile):
        raise AssertionError()
    os.remove(outfile)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
