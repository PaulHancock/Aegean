import os
import sys
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
    import AegeanTools
    return AegeanTools.__version__


reqs = ['numpy>=1.10',
        'scipy>=0.16',
        'astropy>=2.0, <3',
        'pprocess>=0.5',
        'healpy >=1.10']

if sys.version_info < (2, 7):
    reqs.append('lmfit==0.9.1')
else:
    reqs.append('lmfit>=0.9.2')

data_dir = 'AegeanTools/data'

setup(
    name="AegeanTools",
    version=get_version(),
    author="Paul Hancock",
    author_email="Mr.Paul.Hancock@gmail.com",
    description="The Aegean source finding program, and associated tools.",
    # license = "BSD",
    # keywords="example documentation tutorial",
    url="https://github.com/PaulHancock/Aegean",
    long_description=read('README.md'),
    packages=['AegeanTools'],
    install_requires=reqs,
    scripts=['scripts/aegean', 'scripts/BANE', 'scripts/SR6', 'scripts/AeRes', 'scripts/MIMAS'],
    data_files=[('AegeanTools', [os.path.join(data_dir, 'MOC.fits')]) ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'nose']
)
