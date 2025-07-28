#! /usr/bin/env python
"""
Setup for AegeanTools
"""
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def get_version():
    """Get the version number of AegeanTools"""
    version = ""
    with open("AegeanTools/__init__.py", "r") as f:
        while True:
            l = f.readline()
            if l.startswith("__version__"):
                version = l.split("=")[1].strip().replace('"', "")
                break
    return version


setup(version=get_version(), scripts=["scripts/fix_beam.py"])
