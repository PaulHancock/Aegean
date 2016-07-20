Aegean software tools
======

Programs included:
* aegean.py - The Aegean source finding program. All your^ radio astronomy source finding needs in one spiffy program.
* BANE.py - The Background and Noise Estimation tool. For providing high quality background and noise images for use with Aegean, at a small fraction of the cost of a full box-car smooth.
* MIMAS.py - The Multi-resolution Image Masking tool for Aegean Software. For creating image regions which can be used to restrict the source finding of Aegean, to mask fits files, and to create ds9 region files.
* SR6.py - A tool for shrinking and growing fits files, such as those created with BANE.py --compress. Shrinking is done by decimation, growing is done by linear interpolation.

^ - by "your" I mean "my"

Istalling
=====
You can install via pip using `pip install git+https://github.com/PaulHancock/Aegean.git`

Or you can clone or download the repository and then use `python setup.py install`




Status
=====
[Quantified Code](https://www.quantifiedcode.com/) rating [![Code Issues](https://www.quantifiedcode.com/api/v1/project/b0ca0f0d05e943888383528378c1a3e6/badge.svg)](https://www.quantifiedcode.com/app/project/b0ca0f0d05e943888383528378c1a3e6)

