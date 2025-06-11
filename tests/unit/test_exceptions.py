#! /usr/bin/env python
"""
Test exceptions.py
"""
from __future__ import annotations

__author__ = 'Paul Hancock'


def test_exceptions():
    """
    import and raise exceptions
    """
    try:
        from treasure_island.exceptions import AegeanError, AegeanNaNModelError
    except ImportError as e:
        raise AssertionError("Cannot import AegeanTools.exceptions\n"+e.msg)

    if not issubclass(AegeanError, Exception):
        msg = "AegeanError is not an Exception"
        raise AssertionError(msg)
    if not issubclass(AegeanNaNModelError, AegeanError):
        msg = "AegeanNaNModelError is not an AegeanError"
        raise AssertionError(msg)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
