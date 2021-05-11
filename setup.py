#! /usr/bin/env python
"""
Setup for AegeanTools
"""
import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    """Read a file"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
    """Get the version number of AegeanTools"""
    import AegeanTools
    return AegeanTools.__version__


reqs = ['scipy>=0.16',
        'six>=1.11',
        'tqdm>=4',
        'numpy>=1.16',
        'astropy>=2.0',
        'healpy >=1.10',
        'lmfit>=0.9.2']

data_dir = 'AegeanTools/data'

setup(
    name="AegeanTools",
    packages=['AegeanTools'],
    version=get_version(),
    licence='afl-3.0',

    description="The Aegean source finding program, and associated tools.",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author="Paul Hancock",
    author_email="Mr.Paul.Hancock@gmail.com",
    url="https://github.com/PaulHancock/Aegean",

    install_requires=reqs,
    scripts=['scripts/aegean', 'scripts/BANE', 'scripts/SR6', 'scripts/AeRes', 'scripts/MIMAS'],
    data_files=[('AegeanTools', [os.path.join(data_dir, 'MOC.fits')]) ],
    python_requires='>=3.6',
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'nose'],

    keywords=['image processing', 'radioastronomy'],
    classifiers = [
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Astronomy',
    'License :: OSI Approved :: Academic Free License (AFL)',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
