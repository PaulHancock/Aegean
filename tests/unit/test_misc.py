#! /usr/bin/env python
"""
Tests for modules that are too small to warrant their own test suite
"""

from __future__ import annotations

__author__ = "Paul Hancock"


def test_flags():
    """Test that the flags import without errors"""
    import treasure_island.flags

    # use a flag
    if not treasure_island.flags.FITERR > 0:
        msg = "FITERR is not >0"
        raise AssertionError(msg)


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()
