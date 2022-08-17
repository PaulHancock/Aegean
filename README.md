# AegeanTools software package

Programs included:
-  `aegean` - The Aegean source finding program. All your radio astronomy source finding needs in one spiffy program.
- `BANE` - The Background and Noise Estimation tool. For providing high quality background and noise images for use with Aegean, at a small fraction of the cost of a full box-car smooth.
- `MIMAS` - The Multi-resolution Image Masking tool for Aegean Software. For creating image regions which can be used to restrict the source finding of Aegean, to mask fits files, and to create ds9 region files.
- `SR6` - A tool for shrinking and growing fits files, such as those created with `BANE.py --compress`. Shrinking is done by decimation, growing is done by linear interpolation.
- `AeRes` - A tool for adding or subtracting sources from an image - "Aegean Residual". Catalogues must be in Aegean readable format (eg, written by Aegean, modified by user). This can be used to look for missed sources, mis-characterised sources, or for simulating new images.
- `AeReg` - A tool for adjusting an existing catalogue of sources by regrouping or resizing the components - "Aegean Regroup". Uses the same strategies as the Aegean priorized fitting pre-processing.

## Installing

AegeanTools is built and tested on python 3.8-3.10.

You can install via pip using 
`pip install git+https://github.com/PaulHancock/Aegean.git` (latest)
`pip install AegeanTools` (stable)

Or you can clone or download the repository and then use `pip install .`

You can use a pre-built version of AegeanTools from [![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/r/paulhancock/aegean).
Use `docker pull paulhancock/aegean` to pull the image.


## Help
Please see the [wiki pages](https://github.com/PaulHancock/Aegean/wiki) for some help and examples. If you have questions that are not answerd in the wiki please feel free to email me. If you have found a bug or some inconsistency in the code please [submit a ticket](https://github.com/PaulHancock/Aegean/issues) (which will trigger an email to me) and I'll get right on it. 

## Credit

If you use Aegean or any of the AegeanTools for your research please give credit by citing:
- Paper 1, [Hancock et al 2012, MNRAS, 422, 1812](http://adsabs.harvard.edu/abs/2012MNRAS.422.1812H)
- Paper 2, [Hancock et al 2018, PASA, 35, 11H](http://adsabs.harvard.edu/abs/2018PASA...35...11H)

Until there is a more appropriate method for crediting software development and maintainance, please also consider including me as a co-author on publications which rely on Aegean or AegeanTools.


## Status

[![Build Status](https://travis-ci.org/PaulHancock/Aegean.svg?branch=master)](https://travis-ci.org/PaulHancock/Aegean) [![Documentation Status](https://readthedocs.org/projects/aegeantools/badge/?version=latest)](http://aegeantools.readthedocs.io/en/latest/?badge=latest)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7126a16b98804879af2f534e21492b13)](https://www.codacy.com/gh/PaulHancock/Aegean/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=PaulHancock/Aegean&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/7126a16b98804879af2f534e21492b13)](https://www.codacy.com/gh/PaulHancock/Aegean/dashboard?utm_source=github.com&utm_medium=referral&utm_content=PaulHancock/Aegean&utm_campaign=Badge_Coverage)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3474072.svg)](https://doi.org/10.5281/zenodo.3474072)


