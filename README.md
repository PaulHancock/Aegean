Aegean software tools
======

Programs included:
* aegean - The Aegean source finding program. All your^ radio astronomy source finding needs in one spiffy program.
* BANE - The Background and Noise Estimation tool. For providing high quality background and noise images for use with Aegean, at a small fraction of the cost of a full box-car smooth.
* MIMAS - The Multi-resolution Image Masking tool for Aegean Software. For creating image regions which can be used to restrict the source finding of Aegean, to mask fits files, and to create ds9 region files.
* SR6 - A tool for shrinking and growing fits files, such as those created with BANE.py --compress. Shrinking is done by decimation, growing is done by linear interpolation.

^ - by "your" I mean "my"

Istalling
=====
You can install via pip using 
`pip install git+https://github.com/PaulHancock/Aegean.git` (latest)
`pip install AegeanTools` (stable)

Or you can clone or download the repository and then use `python setup.py install`

Help
=====
Please see the [wiki pages](https://github.com/PaulHancock/Aegean/wiki) for some help and examples. If you have questions that are not answerd in the wiki please feel free to email me. If you have found a bug or some inconsistency in the code please [submit a ticket](https://github.com/PaulHancock/Aegean/issues) (which will trigger an email to me) and I'll get right on it. 

Credit
=====
If you use Aegean or any of the AegeanTools for your research please credit me by citing [Hancock et al 2012, MNRAS, 422, 1812](http://adsabs.harvard.edu/abs/2012MNRAS.422.1812H). 

Until there is a more appropriate method for crediting software development and maintainance, please also consider including me as a co-author on publications which rely on Aegean or AegeanTools.


Status
=====
[Quantified Code](https://www.quantifiedcode.com/) rating [![Code Issues](https://www.quantifiedcode.com/api/v1/project/b0ca0f0d05e943888383528378c1a3e6/badge.svg)](https://www.quantifiedcode.com/app/project/b0ca0f0d05e943888383528378c1a3e6)

[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

