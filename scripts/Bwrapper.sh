#! /bin/bash

# usage is:
# Bwrapper infile outfile [--compress]
tempfile='no_one_would_choose_this_name'
BANE.py ${1} --out ${tempfile} --onepass 

cmd="python -c \"from astropy.io import fits; import numpy as np; f=fits.open('${1}'); b=fits.open('${tempfile}_bkg.fits')[0].data; f[0].data -= b; f.writeto('${tempfile}_sub.fits',clobber=True)\""

eval ${cmd}

BANE.py ${tempfile}_sub.fits --out ${2%%.fits} --onepass ${3}
mv ${tempfile}_bkg.fits ${2%%.fits}_bkg.fits
rm ${tempfile}*