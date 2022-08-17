# MIMAS

## Motivation

Prior to 1.8.1, the Aegean source-finding program operated on the entire input image. To return a list of sources that were contained within a sub region other programs were required (For example stilts). Normally this is not a big concern as the filtering process is rather fast. Since radio telescopes have circular primary beam patterns, and fits images are forced to be rectangular, the images produced by imaging pipelines would contain the area of interest along with some amount of extra sky. If the pixels outside the area of interest are not flagged or masked by the imaging pipeline then extra tools are required. Not being able to find any nifty tools to do this job for me, I decided to create the Milti-resolution Image Mask for Aegean Software - MIMAS. There are three main features that I was looking for, each of which are solved by MIMAS.

## Aims

MIMAS was created with the following three goals in mind:

* to be able to create and manipulate arbitrary shaped regions that could be used to describe areas of sky. The method of manipulation is intended to parallel that of set operations so that you can easily take the intersection, union, or difference of regions, in order to create regions as simple as circles and polygons, to some horrendous thing that describes the sky coverage of a survey.

* to be able to store these regions in a file format that can be easily stored and transmitted.

* to be able to use these regions to mask image files, or to restrict the operation of Aegean to a sub section of a given image.

## Methodology

MIMAS is a wrapper script that uses the regions module that is now part of AegeanTools. The regions module contains a suite of unit tests and a single class called Region. The Region class is built on top of the [HealPy](https://github.com/healpy/healpy) module, which is in turn a wrapper around the [HEALPix](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2005ApJ...622..759G&db_key=AST&high=41069202cf02947) software. 


## Usage

MIMAS has five modes of operation:
* create a new region from a combination of: stored regions, circles, or polygons.
* create a new region from a [DS9](http://ds9.si.edu/site/Home.html) .reg file
* convert a region.mim file into a .reg format that can be used as an overlay for DS9.
* use a .fits image to create a .mim region file as if the image were a mask
* use a region file and a .fits image to create a new fits image where pixels that are OUTSIDE the given region have been masked.

The operation of MIMAS is explained by the following help text:
```console
usage: MIMAS [-h] [-o OUTFILE] [-depth N] [+r [filename [filename ...]]] [-r [filename [filename ...]]]
             [+c ra dec radius] [-c ra dec radius] [+p [ra [dec ...]]] [-p [ra [dec ...]]] [-g]
             [--mim2reg region.mim region.reg] [--reg2mim region.reg region.mim] [--mim2fits region.mim region_MOC.fits]
             [--mask2mim mask.fits region.mim] [--intersect region.mim] [--area region.mim]
             [--maskcat region.mim INCAT OUTCAT] [--maskimage region.mim file.fits masked.fits]
             [--fitsmask mask.fits file.fits masked_file.fits] [--negate] [--colnames RA_name DEC_name]
             [--threshold THRESHOLD] [--debug] [--version] [--cite]

optional arguments:
  -h, --help            show this help message and exit

Creating/modifying regions:
  Must specify -o, plus or more [+-][cr]

  -o OUTFILE            output filename
  -depth N              maximum nside=2**N to be used to represent this region. [Default=8]
  +r [filename [filename ...]]
                        add a region specified by the given file (.mim format)
  -r [filename [filename ...]]
                        exclude a region specified by the given file (.mim format)
  +c ra dec radius      add a circle to this region (decimal degrees)
  -c ra dec radius      exclude the given circles from a region
  +p [ra [dec ...]]     add a polygon to this region ( decimal degrees)
  -p [ra [dec ...]]     remove a polygon from this region (decimal degrees)
  -g                    Interpret input coordinates are galactic instead of equatorial.

Using already created regions:
  --mim2reg region.mim region.reg
                        convert region.mim into region.reg
  --reg2mim region.reg region.mim
                        Convert a .reg file into a .mim file
  --mim2fits region.mim region_MOC.fits
                        Convert a .mim file into a MOC.fits file
  --mask2mim mask.fits region.mim
                        Convert a masked image into a region file
  --intersect region.mim, +i region.mim
                        Write out the intersection of the given regions.
  --area region.mim     Report the area of a given region

Masking files with regions:
  --maskcat region.mim INCAT OUTCAT
                        use region.mim as a mask on INCAT, writing OUTCAT
  --maskimage region.mim file.fits masked.fits
                        use region.mim to mask the image file.fits and write masekd.fits
  --fitsmask mask.fits file.fits masked_file.fits
                        Use a fits file as a mask for another fits file. Values of blank/nan/zero are considered to be
                        mask=True.
  --negate              By default all masks will exclude data that are within the given region. Use --negate to exclude
                        data that is outside of the region instead.
  --colnames RA_name DEC_name
                        The name of the columns which contain the RA/DEC data. Default=(ra,dec).

Extra options:
  --threshold THRESHOLD
                        Threshold value for input mask file.
  --debug               debug mode [default=False]
  --version             show program's version number and exit
  --cite                Show citation information.

Regions are added/subtracted in the following order, +r -r +c -c +p -p. This means that you might have to take multiple passes to construct overly complicated regions.
```

### Data model and operation 

At the most basic level, The Regions class takes a description of a sky area, either a circle or a polygon, and converts it into a list of HELAPix pixels. These pixels are stored as a python set, making it easy to implement set operations on these regions. HEALpix is a parameterization of the sky that maps diamond shaped regions of equal area, onto a pixel number. There are many interesting properties of the nested HEALPix parameterization that make it easy to implement the Region class. Firstly, HEALPix can represent areas of sky that are as coarse as 1/12th of the entire sky, to regions that are 1/2^30 times smaller. A depth or resolution parameter of 2^12 represents a pixel size of less than one arcminute. By making use of different resolutions of pixels, it is possible to represent any region in an efficient manner. The sky area that is represented by a Region is a combination of pixels of different resolutions, with the smallest resolution being supplied by the user.

## File format

The MIMAS program is able to take a description of a region and save it to a file for use by many programs. Since he underlying data model is a dictionary of sets, the fastest and easiest file format to use is that given by the cPickle module (a binary file). These files are small, fast to read and write, and accurately reproduce the region object that was stored. The MIMAS program writes files with an extension of .mim.

## Interaction with Aegean

Region files with .mim extension that are created by MIMAS can be used to restrict Aegean to the given region of an image. Use the `--region region.mim` option when running Aegean to enable this.