2018-07-02
==========
General
- Drop requirement for h5py
- Drop support for h5py since it doesn't work

AeRes
- refactored functionality into `AegeanTools/AeRes.py`
- help text is now printed `AeRes` is run without arguments
- fix a bug that caused by comparing `map` to a `float` in python3 (h/t Tom F.)


2018-06-28
==========
BANE
 - multiple changes that allow BANE to be run in memory constrained enviornments
 - force image segmentation to always be in horizontal stripes
 - add new option `--stripes` to control the number of stripes
 - allow `--stripes` and `--cores` to be different
 - make better use of shared memory to reduce memory footprint
 - update BANE to version 1.6.0

v 2.0.2
=======
General
- Convert licence from lGPL to AFL 3.0
- Create a changelog

Aegean
- migrate to argparse
- reformat the `--help` text to what I think is more clear and orderly
- add new command line option `--forcebkg` which forces a single background value to be used over the entire image
- add new command line option `--columnprefix` which adds a prefix to each column of the out-put catalogues

BANE
- die gracefully when killed
- quit gracefully when NAXIS>4

v 2.0.1
=======
General
- update README to include install instructions, citations, and code checking badges

Aegean
- a new flag `flags.PRIORIZED` (=64) is set when a source is fit via priorized fitting

v 2.0 (vs 1.0)
=============
General
- Support python 2.7 and 3.6 (test/build) <2.6 may still work but not tested
- build/test using [TravisCI](travis-ci.org)
- track test coverage with [Coveralls](coveralls.io)
- monitor code quality with [Codacy](codacy.com)
- publish to [PyPi](https://pypi.org/project/AegeanTools/)
- publish to [Zenodo](zenodo.org)
- description paper: [Hancock et al 2018, PASA, 35, 11H](http://adsabs.harvard.edu/abs/2018PASA...35...11H)
- add new programs: BANE, SR6, MIMAS, and AeRes

Aegean
- Use covariance matrix to account for correlated noise in fitting, and for determining parameter uncertainties
- new fitting option `--priorize`, skips the source finding stage and goes directly to characterisation
- deprecated `--measure` in place of `--priorize`
- all the psf to vary across the image either in a simple Dec dependant manner, or via an external map `--psf`
- allow negative sources to be fit using `--negative`
- characterise islands via `--island`
- restrict image search area using MIMAS regions
- fit using multiple cores when available
- enable support for reading a single image from a cube using `--slice`

BANE
- **New Program**: The Background And Noise Estimator
- performs a much better calculation of the background and noise of an image as compared to the Aegean built-in

SR6
- **New Program**: Shrink/expand fits images via decimation/interpolation
- complies with the BANE keyword standards so that `--compress`ed bkg/rms images can be restored to full size

MIMAS
- **New Program**: Multi-resolution Image Masking tool for Aegean Software
- describe regions of sky using HEALPix-els
- create regions from circles and polygons
- combine regions using binary set operators
- write .fits (MOC) and .reg (DS9) versions of regions

AeRes
- **New Program**: subtract a model catalogue from an image and return the residual map
