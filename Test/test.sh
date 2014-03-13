#! /usr/bin/env bash

echo "
Test script for Aegean.
These are commands that have been shown to cause errors in development code
"

tst () {
	echo "testing ${1}"
	echo "${1}" > 'out.log'
	echo "${1}" > 'err.log'
	`${1} 1>>'out.log' 2>>'err.log'`
	code=$?
	if [[ ${code} -ne 0 ]] ; then
		echo "Failed: error log follows"
		cat err.log
		exit ${code}
	else
		echo "Success"
	fi
return ${code}
}

rm out.log err.log
#test positive/negative source funcionality
tst "python aegean.py Test/Images/1904-66_SIN.fits --nonegative"
tst "python aegean.py Test/Images/1904-66_SIN_neg.fits --nopositive --island --table=out.vot"

#save background with single/multiple cores
tst "python aegean.py Test/Images/1904-66_SIN.fits --save --cores=1"
tst "python aegean.py Test/Images/1904-66_SIN.fits --save --cores=2"

#load background and process image
tst "python aegean.py Test/Images/1904-66_SIN.fits --rmsin=aegean-rms.fits --bkgin=aegean-background.fits --out=out.cat"

#create an output catalog in Aegean format
tst "python aegean.py Test/Images/1904-66_SIN.fits --out=out.cat --table=table.xml"

#do forced measurements with this catalog
tst "python aegean.py Test/Images/1904-66_SIN.fits --catalog=out.cat --table=kvis.ann"

#do island fitting and ouput a ds9 reg file
tst "python aegean.py Test/Images/1904-66_SIN.fits --island --table=ds9.reg"

#use a user supplied beam parameter
tst "python aegean.py Test/Images/1904-66_SIN.fits --beam=0.3 0.3 0"

#do island fitting but don't ouput any tables
tst "python aegean.py Test/Images/1904-66_SIN.fits --island --out=out.cat"

#test some hdu options
tst "python aegean.py Test/Images/MultiHDU.fits --hdu=1 --out=out.cat"

