#! /bin/bash

echo "
Test script for BANE.
These are commands that have been shown to cause errors in development code
"

tst () {
	echo "testing ${1}"
	echo "${1}" > 'out.log'
	echo "${1}" > 'err.log'
	`${1} 1>>'out.log' 2>>'err.log'`
	code=$?
	if [ -z "$2" ] ; then
	    c=0
	else
	    c="$2"
	fi
	if [[ ${code} -ne ${c} ]] ; then
		echo "Failed: error log follows"
		cat err.log
		exit ${code}
	else
		echo "Success"
	fi
return ${code}
}

echo "Testing BANE"
rm out.log err.log

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux"

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux --compress"

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux --grid 12 10 --compress"

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux --cores 1"

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux --onepass"

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux --nomask"

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux --noclobber" 1

tst "python BANE.py Test/Images/1904-66_SIN.fits --out aux --grid 12 10 --compress"

tst "python BANE.py Test/Images/MultiHDU.fits --out aux"

echo "to clean up:"
echo "rm aux_{bkg,rms}.fits"


