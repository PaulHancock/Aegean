#! /usr/bin/env python
"""
Tests for modules that are too small to warrant their own test suite
"""

__author__ = "Paul Hancock"


def test_flags():
    """Test that the flags import without errors"""
    import AegeanTools.flags

    # use a flag
    if not AegeanTools.flags.FITERR > 0:
        raise AssertionError("FITERR is not >0")
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
