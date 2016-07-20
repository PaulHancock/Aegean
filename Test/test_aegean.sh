#! /bin/bash

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


echo "Testing Aegean"
# show module versions
tst "aegean --versions"

# test positive/negative source funcionality
tst "aegean Test/Images/1904-66_SIN.fits --negative --nopositive"
tst "aegean Test/Images/1904-66_SIN_neg.fits --negative --island --table=out.vot"

# save background with single/multiple cores
tst "aegean Test/Images/1904-66_SIN.fits --save --cores=1 --beam 0.3 0.3 0"
tst "aegean Test/Images/1904-66_SIN.fits --save --cores=2 --outbase=aux"

# load background and process image
tst "aegean Test/Images/1904-66_SIN.fits --noise=aux_rms.fits --background=aux_bkg.fits --out=out.cat"

# create an output table in various formats
tst "aegean Test/Images/1904-66_SIN.fits --out=out.cat --table=table.xml,table.vot,table.csv,table.tex,table.tab"

# do priorized measurements with an input catalog
tst "aegean Test/Images/1904-66_SIN.fits --input table_comp.vot --priorized 1 --table table_prior.vot"
tst "aegean Test/Images/1904-66_SIN.fits --input table_comp.vot --priorized 1 --table table_prior.vot --ratio 0.9"
tst "aegean Test/Images/1904-66_SIN.fits --input table_comp.vot --priorized 1 --table table_prior.vot --ratio 1.4"

# do island fitting and ouput a ds9 reg file and an sqlite3 database
tst "aegean Test/Images/1904-66_SIN.fits --island --table=ds9.reg,my.db"

# use a user supplied beam parameter
tst "aegean Test/Images/1904-66_SIN.fits --beam=0.3 0.3 0"

# do island fitting but don't ouput any tables
tst "aegean Test/Images/1904-66_SIN.fits --island --out=out.cat"

# test some hdu options
tst "aegean Test/Images/MultiHDU.fits --hdu=1 --out=out.cat"

# test WCS problem handling
tst "aegean Test/Images/WCS_edge.fits"

echo "to clean up:"
echo "rm 1904-66_SIN_{bkg,rms}.fits aux_{bkg,rms,crv}.fits out{_comp,_isle}.vot"
echo "rm out.{cat,db} table_comp.{xml,vot,csv,tex,tab} kvis_simp.ann ds9{_comp,_isle}.reg my.db"
echo "rm table_proir_comp.vot"

# Test on an image that has a very large island in it. Ideally should finish within 30s.
echo "This next test will probably fail because Aegean is slow for large islands."
tst "timeout -sHUP 30s aegean --autoload Test/Images/Bright.fits"




