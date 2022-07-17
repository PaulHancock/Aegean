#! /usr/bin/env python
"""
Test exceptions.py
"""

__author__ = 'Paul Hancock'


def test_exceptions():
    """
    import and raise exceptions
    """
    try:
        from AegeanTools.exceptions import AegeanError, AegeanNaNModelError
    except ImportError as e:
        raise AssertionError("Cannot import AegeanTools.exceptions\n"+e.msg)

    if not issubclass(AegeanError, Exception):
        raise AssertionError("AegeanError is not an Exception")
    if not issubclass(AegeanNaNModelError, AegeanError):
        raise AssertionError("AegeanNaNModelError is not an AegeanError")
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
