#! /usr/bin/env python


__author__ = 'hancock'

import sys
import re
from AegeanTools.catalogs import load_table, table_to_source_list, save_catalog

if __name__ == "__main__":
    infile, outfile = sys.argv[-2:]
    catalog = table_to_source_list(load_table(infile))
    outfile = re.sub('_comp','',outfile)
    save_catalog(outfile,catalog)