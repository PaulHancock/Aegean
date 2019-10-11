#! /usr/bin/env python

"""

"""

from AegeanTools import pprocess
import traceback
import sys

def make_stupid_err(m):
    m**m
    return

def raise_stupid_err(message):
    print('doing bad things')
    make_stupid_err(message)
    return

def test_traceback():
    try:
        raise_stupid_err('one core')
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)


def test_traceback2():

    queue = pprocess.Queue(limit=5, reuse=1)
    do_parallel = queue.manage(pprocess.MakeReusable(raise_stupid_err))

    for i in range(5):
        do_parallel('Process {0}'.format(i))

    try:
        queue.next()
    except TypeError as e:
        if 'pow()' not in repr(e):
            raise AssertionError("Type Error not raised properly")
    return

if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()