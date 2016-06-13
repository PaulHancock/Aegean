import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_version():
    return "1.9.6"

setup(
    name = "AegeanTools",
    version = get_version(),
    author = "Paul Hancock",
    author_email = "Mr.Paul.Hancock@gmail.com",
    description = (""),
#    license = "BSD",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/an_example_pypi_project",
#    packages=['an_example_pypi_project', 'tests'],
    long_description=read('README'),
    packages = ['aegean'],
    install_requires=['numpy>=1.10', 'scipy>=0.16','astropy>=1.0']
    scripts=['scripts/aegean','scripts/BANE','scripts/SR6','scripts/AeRes','scripts/MIMAS']
)