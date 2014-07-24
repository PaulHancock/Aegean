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
#show module versions
tst "python aegean.py --versions"

#test positive/negative source funcionality
tst "python aegean.py Test/Images/1904-66_SIN.fits --negative --nopositive"
tst "python aegean.py Test/Images/1904-66_SIN_neg.fits --negative --island --table=out.vot"

#save background with single/multiple cores
tst "python aegean.py Test/Images/1904-66_SIN.fits --save --cores=1 --beam 0.3 0.3 0"
tst "python aegean.py Test/Images/1904-66_SIN.fits --save --cores=2 --outbase=aux"

#load background and process image
tst "python aegean.py Test/Images/1904-66_SIN.fits --noise=aux_rms.fits --background=aux_bkg.fits --out=out.cat"

#create an output table in various formats
tst "python aegean.py Test/Images/1904-66_SIN.fits --out=out.cat --table=table.xml,table.vo,table.csv,table.tex,table.tab"

#do forced measurements with this catalog
tst "python aegean.py Test/Images/1904-66_SIN.fits --measure --input=out.cat --table=kvis.ann"

#do island fitting and ouput a ds9 reg file and an sqlite3 database
tst "python aegean.py Test/Images/1904-66_SIN.fits --island --table=ds9.reg,my.db"

#use a user supplied beam parameter
tst "python aegean.py Test/Images/1904-66_SIN.fits --beam=0.3 0.3 0"

#do island fitting but don't ouput any tables
tst "python aegean.py Test/Images/1904-66_SIN.fits --island --out=out.cat"

#test some hdu options
tst "python aegean.py Test/Images/MultiHDU.fits --hdu=1 --out=out.cat"

#measure and find at the same time with islands
tst "python aegean.py Test/Images/1904-66_SIN.fits --measure --input=out.cat --find --island --table=out.db"

#test WCS problem handling
#tst "python aegean.py Test/Images/WCS_edge.fits"

echo "to clean up:"
echo "rm 1904-66_SIN_{bkg,rms}.fits aux_{bkg,rms}.fits out{_comp,_isle}.vot out.{cat,db} table_comp.{xml,vo,csv,tex,tab} kvis_simp.ann ds9{_comp,_isle}.reg my.db"





