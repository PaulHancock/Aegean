v 2.1.1
=======
General
- drop support for python <2.7
 
Aegean
- Support for lmfit v 0.9.15 and above (affects priorized fitting only)
- force `psf_pa` to be a float (ie 0.0) when BPA is not supplied
- ensure that `err_*` columns are always reported as floats (ie errs are now -1.0)
- report filename of interest when region files can't be found 

BANE
- `filter_image` will now return maps and saving to a file is optional

SR6
- automatically determine a default reduction factor

v 2.1.0
=======
General
- Begin preparation for a major update (Aegean 3.0)
- Include `pprocess.py` and update it to support python3
- Update testing to include python 3.7 on ubuntu 16.04
- Remove old an unused `Test` directory and files (saves space for installs)

Aegean
- Allow multi-core processing in python3
- Do curvature calculation on islands instead of entire image (small speed improvement)

2019-09-03
==========
Aegean
- Fix a bug that caused int/peak flux ratios to be not 1 in some cases
- Island integrated fluxes are now computed in the same way as components

2018-09-13
==========
BANE
- version is now v1.8.0
- Fixed a segfault that could occur on image of a particular size (usually 4k x 4k)

2018-08-08
==========
SR6
- interpolation is now the same as that performed by BANE and is always linear
- peak memory use is reduced
- deprecated option ``-i MODE`` which selected the interpolation method

2018-08-01
==========
BANE
- version is now v1.7.0
- address a segfault problem that occurs on large images when `--onepass` is not used (#75)
- making use of `--stripes` greater than 1 requires at least `--cores=2`
- further reduce memory footprint:
  - reduce the amount of interpolation stages that are required by 1 per stripe
  - do background subtraction and masking in the sub-processes to avoid loading the entire image in the main process
- deprecate options `--onepass`/`--twopass` and set a warning when used

MIMAS
- fix a bug where `save_region` didn't work in python3

2018-07-19
==========
Aegean
- fix a bug that occured when `--find` and `--priorized` were used together (h/t Stefan D.)

2018-07-05
==========
Aegean
- remove h5py comments from the `--version` output
- fix bug with lmfit version 0.9.11 (issue #70) (h/t James L.)

BANE
- fix a bug that causes only the first stripe to be written to (repeatedly)
- retire some TODO items related to code clarity
- update to v1.6.5

2018-07-02
==========
General
- Drop requirement for h5py
- Drop support for h5py since it doesn't work

AeRes
- refactored functionality into `AegeanTools/AeRes.py`
- help text is now printed `AeRes` is run without arguments
- fix a bug that caused by comparing `map` to a `float` in python3 (h/t Tom F.)


2018-06-28
==========
BANE
 - multiple changes that allow BANE to be run in memory constrained enviornments
 - force image segmentation to always be in horizontal stripes
 - add new option `--stripes` to control the number of stripes
 - allow `--stripes` and `--cores` to be different
 - make better use of shared memory to reduce memory footprint
 - update BANE to version 1.6.0

v 2.0.2
=======
General
- Convert licence from lGPL to AFL 3.0
- Create a changelog

Aegean
- migrate to argparse
- reformat the `--help` text to what I think is more clear and orderly
- add new command line option `--forcebkg` which forces a single background value to be used over the entire image
- add new command line option `--columnprefix` which adds a prefix to each column of the out-put catalogues

BANE
- die gracefully when killed
- quit gracefully when NAXIS>4

v 2.0.1
=======
General
- update README to include install instructions, citations, and code checking badges

Aegean
- a new flag `flags.PRIORIZED` (=64) is set when a source is fit via priorized fitting

v 2.0
================
General
- Support python 2.7 and 3.6 (test/build) <2.6 may still work but not tested
- build/test using [TravisCI](travis-ci.org)
- track test coverage with [Coveralls](coveralls.io)
- monitor code quality with [Codacy](codacy.com)
- publish to [PyPi](https://pypi.org/project/AegeanTools/)
- publish to [Zenodo](zenodo.org)
- description paper: [Hancock et al 2018, PASA, 35, 11H](http://adsabs.harvard.edu/abs/2018PASA...35...11H)
- add new programs: BANE, SR6, MIMAS, and AeRes

Aegean
- Use covariance matrix to account for correlated noise in fitting, and for determining parameter uncertainties
- new fitting option `--priorize`, skips the source finding stage and goes directly to characterisation
- deprecated `--measure` in place of `--priorize`
- all the psf to vary across the image either in a simple Dec dependant manner, or via an external map `--psf`
- allow negative sources to be fit using `--negative`
- characterise islands via `--island`
- restrict image search area using MIMAS regions
- fit using multiple cores when available
- enable support for reading a single image from a cube using `--slice`


### v1.9.7-0-g053427d [ Feb 15 2016]
* Aegean
  * LMFIT version 0.9.x now works without causing a crash (recommend that you use lmfit v0.9.x)
  * integrated flux for islands is now calculated in Jy instead of Jy/Beam/pix
  * `--priorized` no longer does island fitting even when requested.
  * `--out` no longer reports 2x sources when `--priorized` is set
  * the psf (local beam) parameters are now reported by default for output sources.
    * psf will be (0,0,0) if no beam can be determined (and integrated flux will be zero)
  * if a region file is supplied but cannot be used Aegean will now complain

* BANE
 * `pprocess` has been replaced with `mutliprocessing` - runtime is faster.
    * contact me if you used BANE on windows and it is just now breaking.
  * pixels are now subjected to sigma clipping before background/rms are calculated.
    * Three rounds of 3 sigma
    * speed is greatly improved
    * accuracy is imporoved
  * running a moderately sized image, on a machine with N cores, will no longer cause a crash when you set `--cores=N-1`

* MIMAS
  * now option `--fitsimage` added.
    * the output is a fits [[MOC|http://ivoa.net/documents/MOC/]] use Aladin to read it.
  * new option `-g` will cause all coordinates to be interpreted as galactic l/b instead of ra/dec

* AeRes
  * NEW program. The aim of AeRes is to produce residual images for Aegean.
  * AeRes will take an input catalog, and an input image, and produce a residual image, and an optional model image.
    * Catalog can be any format that Aegean writes.
  * testing is still underway - comments and feedback are welcome.

### v1.9.6-103-gb9b0e73 [Oct 02 2015]
* Aegean
  * Fixed a bug where the beam size was incorrectly calculated
  * New option `--psf` allows the user to supply a fits file that describes the PSF as a function of sky position
    * The psf file needs to have 5 dimensions: RA, DEC, BMAJ, BMIN, BPA
    * The projection used in the psf file can be different from the input image file.
  * `--autoload` updated to look for a file called `<imname>_psf.fits` and use that as the input to `--psf`

  * `--priorized` fitting has been updated
    * `--catpsf=CATPSF` allows the user to supply an image which describes the psf of the input catalog. This will allow Aegean to deconvolve the sources in the input catalog, and (re) convolve the sources with the image psf to get the correct source sizes.
    * Sources that have no known psf are excluded from the fitting.
    * When position/shape is not fit, the errors are copied from the input catalog. Note that these are copied, and not modified, even though some scaling may be appropriate.
    * Regrouping of islands is now a lot faster than before. It is also more consistent.
    * Priorized fitting uses the same fitting techniques as Aegeans blind source finding. This includes the use of an inverse covariance matrix.

* BANE
  * Grid/Box size is now 16/96 when beam size is unknown.
  * Images with greater than 2 axes no longer cause BANE to crash
  * Interpolation is now done within sub-processes so this step is faster than before.
  * Priorized fitting uses the same fitting techniques as Aegeans blind source finding. This includes the use of an inverse covariance matrix.

### v1.9.5-56-g662ea65 [May 20 2015]
* Aegean
  * Fitting now uses an inverse covariance matrix to account for the correlated nature of radio images.
  * Fitting errors have been updated to reflect the above change. They are similar to Condon'97 errors, but they should not be expected to be the same.
  * Fitting now uses an analytical Jacobian function which gives an increase in the fitting speed.
  * `--priorized` fitting is now done in parallel using `--cores` in the same way that regular fitting is done.
* MIMAS
  * New option `--mim2fits` will now output a correctly formatted Milti-Order Coverage map (MOC) according to [[IVOA recommendation|http://ivoa.net/documents/MOC/]]. These footprint files can be visualised by Aladin. Reading of these fits files has not **yet** been implemented in any of the Aegean tools.
* BANE
  * The peak memory usage has been reduced. Each process that is spawned by BANE will now only read about 1/nth of the total image. The process overhead is about 20Mb on my own machine, but your results may vary.

### v1.9.5-0-gd86d0e9 [May 14 2015]
* Aegean
 * Least Squares fitting is now done via module [[lmfit|http://lmfit.github.io/lmfit-py/]] which wraps ``scipy.optimize`` functions, which in turn rely on ``MINPACK`` code written in c/fortran. ``lmfit`` can be installed with ``pip`` and is faster than ``mpfit``.
 * New option ``--priorized`` has been implemented. This should be used in place of ``--measure``. See [[Priorized Fitting|Priorized Fitting]] for details on how this option works.
 * Catalogs of sources now include a ``uuid`` column that will make exact matching easier for ``--priorized`` fitting.
 * Added new output formats:
   * Fits binary tables with extension ``.fits``.
    * HDF5 tables with extension ``.hdf5`` (requires module ``h5py``)
 * New options ``--tformats`` prints a list of file formats that are avilable for writing. [H/T David Kaplan for this suggestion and the new formats].
 * Updated kvis component annotations file to include an (isle,source) label as is done in the ds9 region files. [Thanks to Shane O'Sullivan for the suggestion and formatting]
* AegeanTools
  * A lot of code has been moved out of ``aegean.py`` and into modules within the ``AegeanTools`` directory. If you are used to importing Aegean to have access to some functionality, you should now be able to import the same functionality from a smaller module within ``AegeanTools``.
  * I have put effort into keeping these modules as independent as possible so that they should be able to be easy to re-use.


###  v1.9rc1-156-g1b33813 [Mar 13 2015]

* Aegean
  * The reported errors now take into account the correlated nature of the data
    * This is based on the description given in the last section of Condon'97
* MIMAS
  * Fixed a bug with the +/- p command line options (H/T Bradley)

###  v1.9rc1-132-g8c0e7af [Mar 2 2015]
* Aegean
  * Errors that are reported are now calculations based on Condon'97
    * These errors assume that the model is a good description of the data
  * Two additional parameters are reported in the output tables, which describe the goodness of fit
    * `residual_mean` is the mean value of (model-data)/noise, and `residual_std` is the standard deviation of this same quantity.
    * These measures may be adjusted to be more useful in the future. Feedback is welcome.
* BANE
  * If `--out` is not specified the output file names will now be based on the input filename. This is instead of the default out_{bkg,rms}.fits names that were previously used
  * Added option `--noclobber` which will halt BANE if the intended output files already exist. The intention is to avoid having to reprocess files that already exist. Without this option BANE will just overwrite the old files.


### v1.9rc1-114-g337ef50 [Feb 17 2015]
* Aegean
  * sources that are not fit now have better major/minor properties so that int_flux/peak_flux is again 1:1
  * forced measurements don't use elevation scaling but a warning will be given if it is requested
  * island fluxes now include elevation scaling when `--telescope` or `--lat` are supplied
* BANE
  * Fixed a bug that would sometimes insert a dodgy pixel into background or rms images (ie rms<0), into the cantral regions of an image.
    * I estimate that the number of such dodgy pixels is always less than the number of cores that you use to process the data.


### v1.9rc1-110-g3627a8a [Feb 6 2015]

* Aegean
  * The beam area used to calculate integrated fluxes can now be scaled by `cos(dec-lat)` where `lat` is the latitude of the observing telescope.
  * This is only required for some projections (like ZEA) but not others (like SIN), so be smart when using this option
  * use `--lat <float>` to set the latitude of the telescope
  * use `--telescope <name>` to let Aegean lookup the latitude from a list of known telescopes
    * Aegean currently knows about MWA|ATCA|VLA|LOFAR

### v1.9rc1-104-gbfe836c [Feb 4 2015]

* SR6 - NEW!
 * Shrink Ray-6 is a command line wrapper for the new `AegeanTools.fits_interp` module
 * `SR6.py <inputfile> -f 10 -o <outputfile>` will decimated a file by sampling every 10th pixel
 * `SR6.py <inputfile> -x -o <outputfile>` will expand a file to full resolution using information stored in the fits header
 * `-m <filename>` will use the given file as a model to mask the output. Pixels which are masked in the model file will be masked in the output file. WCS is ignored, but pixel dimensions must be identical.
 * `-i [linear|nearest|cubic]` controls the interpolation method used. *linear* is default (and recommended), whilst *cubic* is unstable and time consuming.
 * `fits_interp` handles the compression/expansion of fits files using decimation and interpolation
   * when combined with BANE and Aegean `fits_interp` allows for a great reduction in the size of the background and noise images.

* BANE
 * added option `--compress` which will cause BANE to write background an noise images that are low resolution. The lower resolution does not destroy information as the previous 'full-resolution' maps are just interpolated versions of these lower resolution images.
   * `--compress`-ed images are smaller by a factor of about grid^2
    * `--compress`-ed images have correct WCS and can thus be used by any program that you would normally use
    * `--compress`-ed images have key words added to the fits header so that they can be expanded to 'full-resolution' images correctly
 * added option `--nomask` that will stop BANE from masking the output image to match the input image


* Aegean
 * `fits_image` has been modified to automatically detect files that have been created with `--compress` and will expand/interpolate them before loading.

### v1.9rc1-84-g63325f4 [Feb 2 2015]

* Aegean
 * fixed a bug that was stopping negative sources from being found
 * fixed a bug that caused a crash when a fit failed

### v1.9rc1-80-gf3cc16f [Jan 21 2015]

* Aegean
 * Errors on all parameters have now been more correctly calculated. In most cases the errors will decrease. Error calculation is now inline with what the extensive doc string of `mpfit.py` suggests.
 * added new command line option `--autoload`
   * `--autoload` will look for background (_bkg.fits), noise (_rms.fits), and region (.mim) files that have the same base name as the input image. If found these files will be loaded, if not, Aegean proceeds as if you had not specified them.
    * `--autoload` is handy, but can be dangerous as it will not complain about missing files

### v1.9rc1-78-gc4461b1 [Jan 19 2015]

* Aegean
  * The shape of the synthesised beam, in pixels, is now correctly calculated and no longer has a spurious scaling with declination (elevation).
  * As a result of the above, the ratio `int_flux/peak_flux` no longer has a declination dependent offset from 1.
  * The ratio of `int_flux/peak_flux` has an average which is about 1.05 for MWA images. This ratio is expected to be 1.00, but the cause of the bias has not yet been identified.

### v1.9rc1-70-gbcd531d [Jan 16 2015]

* Aegean
 * Sources near the edge of the image (the edge of the valid WCS area of the image) are no longer cuasing crazy problems.

### v1.9rc1-68-g1ad3d25 [Jan 14 2015]

* Aegean
 * `find_sources_in_image` now accepts mask=region as either a filename or an instance of `AegeanTools.regions.Region`

### v1.9rc1-66-g2fdde1a [Jan 13 2015]

* Aegean
 * VOTables, databases, and .reg files now contain metadata
  * Currently this is just the Aegean version

### Earlier versions (prior to GitHub) are [[here|SVN-changelog]]
