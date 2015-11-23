#!/bin/sh
# command line invocations that create the files within this folder
python aegean.py Test/Images/1904-66_SIN.fits --autoload --table 1904_auto.vot
python aegean.py Test/Images/Bright.fits --autoload --seedclip 500 --floodclip 100 --table Bright_s500_f100.vot
