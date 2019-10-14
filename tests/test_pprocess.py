#! /usr/bin/env python

"""

"""

from AegeanTools import pprocess
import numpy as np
import traceback
import sys
import time

N = 5
delay = 0.5

def calculate(i, j):
    """
    A supposedly time-consuming calculation on 'results' using 'i' and 'j'.
    """
    time.sleep(delay)
    return  (i*N + j ) *2

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


def test_calc_queue():
    """
    Test a queue which creates list
    """
    queue = pprocess.Queue(limit=4)
    calc = queue.manage(pprocess.MakeParallel(calculate))
    #results = list(range(N*N))
    answers = list(range(0,2*N*N,2))

    for i in range(0, N):
        for j in range(0, N):
            calc(i, j)

    results = sorted([q for q in queue])

    if not np.all(results == answers):
        print(results)
        print(answers)
        raise AssertionError("Calc queue failed to give correct results")


def test_calc_map():
    """
    Test a map which creates a list
    """
    pmap = pprocess.Map(limit=4)
    calc = pmap.manage(pprocess.MakeParallel(calculate))
    answers = list(range(0,2*N*N,2))

    for i in range(0,N):
        for j in range(0,N):
            calc(i,j)

    results = sorted([m for m in pmap])

    if not np.all(results == answers):
        raise AssertionError("Calc map failed to give correct results")


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()