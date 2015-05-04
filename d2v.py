#! /usr/bin/env python


"""
Convert a date format into a version number.

Paul Hancock
3/3/2015
"""

__author__ = "Paul Hancock"

__version__ = "v1.0"
__date__ = '<date>'

import sys

if sys.argv[-1] == __file__:
    print "Usage: version.py Mon Mar 2 17:12:37 2015 +0800"
    sys.exit(1)

tinfo = sys.argv[-5:]

months=['','Jan','Feb','Mar','Apr','May','Jun',"Jul",'Aug','Sep','Oct','Nov','Dec']

month = months.index(tinfo[0])

print "{0}-{1:02d}-{2:02d}".format(tinfo[3],month,int(tinfo[1]))