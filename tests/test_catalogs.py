#! python
from __future__ import print_function

__author__ = 'Paul Hancock'
__date__ = ''

from AegeanTools import catalogs as cat
from AegeanTools.models import OutputSource
import os

import logging
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")


def test_check_table_formats():
    files = ','.join(['a.csv', 'a.fits', 'a.vot', 'a.hdf5',  'a.ann', 'a.docx', 'a'])
    assert not cat.check_table_formats(files)
    assert cat.check_table_formats('files.fits')


def test_show_formats():
    cat.show_formats()


def test_get_table_formats():
    formats = cat.get_table_formats()
    for f in formats:
        name = 'a.'+f
        assert cat.check_table_formats(name)


def test_update_meta_data():
    meta = None
    meta = cat.update_meta_data(meta)
    assert 'PROGRAM' in meta
    meta = {'DATE': 1}
    meta = cat.update_meta_data(meta)
    assert meta['DATE'] == 1


def test_load_save_catalog():
    catalog = [OutputSource()]
    for ext in ['csv', 'vot']:
        fout = 'a.'+ext
        cat.save_catalog(fout, catalog, meta=None)
        fout = 'a_comp.'+ext
        print(fout)
        assert os.path.exists(fout)
        catin = cat.load_catalog(fout)
        assert len(catin) == len(catalog)
        os.remove(fout)


def test_load_table():
    catalog = [OutputSource()]
    for fmt in ['csv', 'vo']:
        fout = 'a.'+fmt
        cat.save_catalog(fout, catalog, meta=None)
        fout = 'a_comp.'+fmt
        tab = cat.load_table(fout)
        assert len(tab) == len(catalog)
        os.remove(fout)



if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")