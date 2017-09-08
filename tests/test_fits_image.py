#! python
from __future__ import print_function

from AegeanTools import fits_image as fi

__author__ = 'Paul Hancock'
__date__ = ''


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")