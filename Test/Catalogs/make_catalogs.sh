# command line invocations that create the files within this folder
python aegean.py Test/Images/1904-66_SIN.fits --autoload --out 1904_auto.aeg
python aegean.py Test/Images/Bright.fits --autoload --seedclip 500 --floodclip 100 --out Bright_s500_f100.aeg
