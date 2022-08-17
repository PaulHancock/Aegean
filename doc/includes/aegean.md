# aegean

## Simple usage
Suggested basic usage (with mostly default parameters):

`aegean RadioImage.fits --table=Catalog.fits`

Usage and short description can be obtained via `aegean`, which is replicated below.

```console
This is Aegean 2.3.0-(2022-08-17)
usage: aegean [-h] [--find] [--hdu HDU_INDEX] [--beam BEAM BEAM BEAM] [--slice SLICE] [--progress] [--forcerms RMS]
              [--forcebkg BKG] [--cores CORES] [--noise NOISEIMG] [--background BACKGROUNDIMG] [--psf IMGPSF]
              [--autoload] [--out OUTFILE] [--table TABLES] [--tformats] [--blankout] [--colprefix COLUMN_PREFIX]
              [--maxsummits MAX_SUMMITS] [--seedclip INNERCLIP] [--floodclip OUTERCLIP] [--island] [--nopositive]
              [--negative] [--region REGION] [--nocov] [--priorized PRIORIZED] [--ratio RATIO] [--noregroup]
              [--input INPUT] [--catpsf CATPSF] [--regroup-eps REGROUP_EPS] [--save] [--outbase OUTBASE] [--debug]
              [--versions] [--cite]
              [image]

positional arguments:
  image

optional arguments:
  -h, --help            show this help message and exit

Configuration Options:
  --find                Source finding mode. [default: true, unless --save or --measure are selected]
  --hdu HDU_INDEX       HDU index (0-based) for cubes with multiple images in extensions. [default: 0]
  --beam BEAM BEAM BEAM
                        The beam parameters to be used is "--beam major minor pa" all in degrees. [default: read from
                        fits header].
  --slice SLICE         If the input data is a cube, then this slice will determine the array index of the image which
                        will be processed by aegean
  --progress            Provide a progress bar as islands are being fit. [default: False]
  --cores CORES         Number of CPU cores to use when calculating background and rms images [default: all cores]

Input Options:
  --forcerms RMS        Assume a single image noise of rms. [default: None]
  --forcebkg BKG        Assume a single image background of bkg. [default: None]
  --noise NOISEIMG      A .fits file that represents the image noise (rms), created from Aegean with --save or BANE.
                        [default: none]
  --background BACKGROUNDIMG
                        A .fits file that represents the background level, created from Aegean with --save or BANE.
                        [default: none]
  --psf IMGPSF          A .fits file that represents the local PSF.
  --autoload            Automatically look for background, noise, region, and psf files using the input filename as a
                        hint. [default: don't do this]

Output Options:
  --out OUTFILE         Destination of Aegean catalog output. [default: No output]
  --table TABLES        Additional table outputs, format inferred from extension. [default: none]
  --tformats            Show a list of table formats supported in this install, and their extensions
  --blankout            Create a blanked output image. [Only works if cores=1].
  --colprefix COLUMN_PREFIX
                        Prepend each column name with "prefix_". [Default = prepend nothing]

Source finding/fitting configuration options:
  --maxsummits MAX_SUMMITS
                        If more than *maxsummits* summits are detected in an island, no fitting is done, only
                        estimation. [default: no limit]
  --seedclip INNERCLIP  The clipping value (in sigmas) for seeding islands. [default: 5]
  --floodclip OUTERCLIP
                        The clipping value (in sigmas) for growing islands. [default: 4]
  --island              Also calculate the island flux in addition to the individual components. [default: false]
  --nopositive          Don't report sources with positive fluxes. [default: false]
  --negative            Report sources with negative fluxes. [default: false]
  --region REGION       Use this regions file to restrict source finding in this image. Use MIMAS region (.mim) files.
  --nocov               Don't use the covariance of the data in the fitting proccess. [Default = False]

Priorized Fitting config options:
  in addition to the above source fitting options

  --priorized PRIORIZED
                        Enable priorized fitting level n=[1,2,3]. 1=fit flux, 2=fit flux/position, 3=fit
                        flux/position/shape. See the GitHub wiki for more details.
  --ratio RATIO         The ratio of synthesized beam sizes (image psf / input catalog psf). For use with priorized.
  --noregroup           Do not regroup islands before priorized fitting
  --input INPUT         If --priorized is used, this gives the filename for a catalog of locations at which fluxes will
                        be measured.
  --catpsf CATPSF       A psf map corresponding to the input catalog. This will allow for the correct resizing of
                        sources when the catalog and image psfs differ
  --regroup-eps REGROUP_EPS
                        The size in arcminutes that is used to regroup nearby components into a single set of components
                        that will be solved for simultaneously

Extra options:
  --save                Enable the saving of the background and noise images. Sets --find to false. [default: false]
  --outbase OUTBASE     If --save is True, then this specifies the base name of the background and noise images.
                        [default: inferred from input image]
  --debug               Enable debug mode. [default: false]
  --versions            Show the file versions of relevant modules. [default: false]
  --cite                Show citation information.
```

### Example usage:
The following commands can be run from the Aegean directory right out of the box, since they use the test images that are included with Aegean.

* Blind source finding on a test image and report results to stdout
  * `aegean tests/test_files/1904-66_SIN.fits`
* As above but put the results into a text file
  * `aegean tests/test_files1904-66_SIN.fits --table out.csv`
  * The above creates a file `out_comp.csv` for the components that were fit
* Do source finding using a catalog input as the initial parameters for the sources
  * `aegean --priorized 1 --input out_comp.csv tests/test_files/1904-66_SIN.fits`
* Source-find an image and save results to multiple tables
  * `aegean --table catalog.csv,catalog.vot,catalog.fits tests/test_files1904-66_SIN.fits`
* Source-find an image and report the components and islands that were found
  * `aegean --table catalog.vot --island tests/test_files1904-66_SIN.fits`
  * The above creates two files: `catalog_comp.vot` for the components, and `catalog_isle.vot` for the islands. The island column of the components maps to the island column of the islands.
* Source-find a sub-region of an image
  * `aegean --region=region.mim tests/test_files1904-66_SIN.fits`
  * The `region.mim` is a region file in the format created by [MIMAS](./MIMAS)

## Output formats
Aegean supports a number of output formats. There is the Aegean default, which is a set of columns separated by spaces, with header lines starting with #. The format is described within the output file itself.

The Aegean default output (which goes to STDOUT) does not contain all of the columns listed below.
Tables created with the `--table` option contain all the following columns, and as much meta-data as I can manage to pack in.

### Table description
Columns included in output tables have the following columns:
* island - numerical indication of the island from which the source was fitted
* source - source number within that island
* background - background flux density in Jy/beam
* local_rms - local rms in Jy/beam
* ra_str - RA J2000 sexigessimal format
* dec_str - dec J2000 sexigessimal format
* ra - RA in degrees
* err_ra - source-finding fitting error on RA in degrees
* dec - dec in degrees
* err_dec - source-finding fitting error on dec in degrees
* peak_flux - peak flux density in Jy/beam
* err_peak_flux - source-finding fitting error on peak flux density in Jy/beam
* int_flux - integrated flux density in Jy. This is calculated from a/b/peak_flux and the synthesized beam size. It is not fit directly.
* err_int_flux - source-finding fitting error on integrated flux density in Jy
* a - fitted semi-major axis in arcsec
* err_a - error on fitted semi-major axis in arcsec
* b - fitted semi-minor axis in arcsec
* err_b- error on fitted semi-minor axis in arcsec
* pa - fitted position angle in degrees
* err_pa - error on fitted position angle in degrees
* flags - fitting flags (should be all 0 for a good fit)
* residual_mean - mean of the residual flux remaining in the island after fitted Gaussian is subtracted
* residual_std - standard deviation of the residual flux remaining in the island after fitted Gaussian is subtracted
* uuid - a universally unique identifier for this component.
* psf_a - the semi-major axis of the point spread function at this location (arcsec)
* psf_b - the semi-minor axis of the point spread function at this location (arcsec)
* psf_pa - the position angle of the point spread function at this location (arcsec)

An island source will have the following columns:
* island - numerical indication of the island
* components - the number of components within this island
* background - background flux density in Jy/beam
* local_rms - local rms in Jy/beam
* ra_str - RA J2000 sexigessimal format
* dec_str - dec J2000 sexigessimal format
* ra - RA in degrees, of the brightest pixel in the island
* dec - dec in degrees, of the brightest pixel in the island
* peak_flux - peak flux density in Jy/beam, of the brightest pixel in the island
* int_flux - integrated flux density in Jy. Computed by summing pixels in the island, and dividing by the synthesized beam size.
* err_int_flux - Error in the above. Currently Null/None since I don't know how to calculate it.
* eta - a correction factor for int_flux that is meant to account for the flux that was not included because it was below the clipping limit. For a point source the true flux should be int_flux/eta. For extended sources this isn't always the case so use with caution.
* x_width - the extent of the island in the first pixel dimension, in pixels
* y_width - the extent of the island in the second pixel dimension, in pixels
* max_angular_size - the largest distance between to points on the boundary of the island, in degrees.
* pa - the position angle of the max_angular_size line
* pixels - the number of pixels within the island
* beam_area - the area of the synthesized beam (psf) in deg^2
* area - the area of the island in deg^2
* flags - fitting flags (should be all 0 for a good fit)
* uuid - a universally unique identifier for this island.

**Note**: Column names with 'ra/dec' will be replaced with a 'lat/lon' version if the input image has galactic coordinates in the WCS.

### Table Types
The most useful output is to use tables. Table output is supported by sqlite and [astropy](https://astropy.org) and there are three main types: database, votable, and ascii table. Additionally you can output ds9 region files by specifying a .reg file extension.

### Database:
This format requires that the sqlite module is available. This is nearly always true by default, but if you get a crash then check that you can `import sqlite3` from a python terminal before submitting a bug report.

Use `--table out.db` to create a database file containing one table for each source type that was discovered. The table names are 'components', 'islands', and 'simples'. Islands are created when --island is enabled. Components are elliptical gaussian fits and are the default type of source to create. Simples are sources that have been created by using the --measure option.

The columns of the database are self explanatory though they have no units. All fluxes are in Jy, major and minor axes are in arcseconds, and the position angle is in degrees. Errors that would normally be reported as -1 in other formats are stored as nulls in the database tables.


### VOTable:
VOTables are difficult to work with as a human, but super awesome to work with when you have [TopCat](http://www.star.bris.ac.uk/~mbt/topcat/) or some other VO enabled software.

VOTable output is supported by AstroPy (0.3+ I think). If you don't have the right version of AstroPy you can still run Aegean but will not be able to write VOTables. You will be told this when Aegean runs.

Use `--table out.vot` or `--table out.xml` to create a VOTable. Each type of sources that you find will be saved to a different file. Components are saved to out_comp.vot, islands are saved to out_isle.vot, and simple sources will be saved to out_simp.vot (or xml as appropriate). See above for a description of the source types.


### ASCII tables:
ASCII tables are supported by AstroPy (0.4+ I think). As with VOTables, if you don't have the right version of AstroPy then Aegean will still run but it will tell you that you can't write ASCII tables.

There are currently four types of ascii tables that can be used:
* csv -> comma separated values
* tab -> tab separated values
* tex -> LaTeX formatted table
* html -> an html formatted table

Use `--table out.html,out.tex` etc.. for the type of table you are interested in. All tables have column headers that are the same as the variable names. These should be easily discernible. The units are Jy for fluxes, arcseconds for major/minor axes, and degrees for position angles.

As with other table formats the file names will be modified to out_comp.html, out_simp.csv, etc... to denote the different types of sources that are contained within.

### FITS binary tables
use extension `fits` or `FITS` (but not `fit` or `FIT`) to write output tables.
Functionality supported by AstroPy.
These are binary tables and only the header is human readable.

### DS9 region files
Use extension `reg` for the output table to get `DS9` region files.
Both components and islands are supported in this format with `_comp.reg` and `_isle.reg` being the corresponding filenames.

Component sources in the `_comp.reg` files will be shown as ellipses at the location of each component, with the fitted size/orientation. Each ellipse will be annotated with the island and component number such that Island 10, component 0 will appear as `(10,0)`.

Island sources will appear as an outline of the pixels that comprise the island. Each island also has an annotation of the island number, and a diagonal line that represents the largest angular scale.

### Flags
There are six different flags that can be set by Aegean during the source finding and fitting process.
In the `STDOUT` version of the Aegean catalog the flags column is written in binary format with a header that read ZWNCPES. These six flags correspond to:


| Abbreviation | Name          | Numerical value | description                                                                                                                                                                                                            |
| ------------ | ------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S            | FITERRSMAL    | 1               | This flag is set when islands are not able to be fit due to there being fewer pixels than free parameters.                                                                                                             |
| E            | FITERR        | 2               | This flag is set when an error occurs during the fitting process. eg the fit doesn't converge.                                                                                                                         |
| P            | FIXED2PSF     | 4               | If a component is forced to have the shape of the local point spread function then this flag is set. This flag is often set at the same time as the FITERRSMALL, or FIXEDCRICULAR                                      |
| C            | FIXEDCRICULAR | 8               | If a source is forced to have a circular shape then this flag will be fit.                                                                                                                                             |
| N            | NOTFIT        | 16              | If a component is not fit then this flag is set. This can because and island has reached the `--maxsummits` limit, or `--measure` mode has been invoked.                                                               |
| W            | WCSERR        | 32              | If the conversion from pixel to sky coordinates doesn't work then this flag will be set. This can happen for strange projections, but more likely when an image contains pixles that don't have valid sky coordinates. |
| Z            | PRIORIZED     | 64              | This flag is set when the source was fit using priorized fitting.                                                                                                                                                      |

Note that the flags column will be the summation of the numerical value of the above flags. So flags=7 means that flags P, E, and S have been set. This all makes more sense when you print the flags in binary format.

## Priorized fitting
This functionality is designed to take an input catalog of sources (previously created by Aegean), and use the source positions and morphologies to measure the flux of these sources within an image.


When ``--priorized x`` is invoked the following will happen:
* input catalog is read from the file specified by ``--input``. This file needs to contain all the properties of a source, including island numbers and uuids. The easiest way to make these files is to just take the output from Aegean and modify it as needed.
* The sources within the catalog are regrouped. The regrouping will recreate islands of sources based on their positions and morphologies. Sources will be grouped together if they overlap at the FHWM. Note that this is different from the default island grouping that Aegean does, which is based on pixels within an island. If ``--noregroup`` is set then the island grouping will be based on the (isle,source) id's in the input catalog.
* Fitting will be done on a per island basis, with multiple sources being fit at the same time. The user is able to control which parameters are allowed to vary at this stage by supplying a number x to ``--priorized x``.
* Fitting will be done on all pixels that are greater than the `--floodclip` limit. If an island has no pixels above this limit then no output source will be generated. Note the special case of `--floodclip -1` which will simply use all pixels within some rectangular region around each input source.
* Output will be written to files as specified by ``--table``.

The parameters that are free/fixed in the fitting process depends on the 'level' of priorized fitting that is requested. Level:

1. Only the flux is allowed to vary. Use this option where you would have otherwise used ``--measure``.
2. Flux and positions are allowed to vary, shape is fixed.
3. Everything is allowed to vary.

In the case that the psf of the input catalogue and the supplied image are different there are three options for describing this difference:
 
1. Use the `--ratio` option, which specifies the ratio of major axes (image psf / catalogue psf). This method works well for small images where the psf doesn't really change over the image, or when the difference is small. 
2. Supply a psf map for the input catalogue using the `--catpsf` option. This will give you ultimate fine control over what the psf of your input catalogue is.
3. Include the psf parameters in the input catalogue as columns `psf_a, psf_b, psf_pa`

Note: If you know how to perform the deconvolve-convolve step for two synthesized beams that are not simply scaled versions of each other, then please let me know so that I can implement this.


### Notes on input tables:
Any [[format|Output-Formats]] that Aegean can write, is an acceptable input format.
The easiest way to create an input table is to modify and existing catalogue.
The following columns are used for priorized fitting:

- Required:  
  - `ra`, `dec`, `peak_flux`, `a`, `b`, `pa`
- Optional: 
  - `psf_a`, `psf_b`, `psf_pa` used for re-scaling the source shapes.
  - `uuid` copied from input to output catalogues
  - `err_ra`, `err_dec` copied from input to output catalogues when positions are not being fit
  - `err_a`, `err_b`, `err_pa` copied from input to output catalogues when shapes are not being fit

Parameters `a`, `b`, `err_a`, `err_b`, `psf_a`, and `psf_b` all have units of arcsec.
Parameters `ra`, `dec`, `pa`,`err_ra`, `err_dec`, and `err_pa` all have units of degrees.