#!/bin/sh
# command line invocations that create the files within this folder
python aegean.py Test/Images/1904-66_SIN.fits --background Test/Images/1904-66_SIN_bkg.fits --noise Test/Images/1904-66_SIN_rms.fits --table 1904_auto.vot
python aegean.py Test/Images/Bright.fits --background Test/Images/1904-66_SIN_bkg.fits --noise Test/Images/1904-66_SIN_rms.fits --seedclip 500 --floodclip 100 --table Bright_s500_f100.vot
