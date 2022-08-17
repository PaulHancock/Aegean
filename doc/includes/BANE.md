# BANE

## Motivation

Aegean has an inbuilt background and noise calculation algorithm (the zones algorithm) which is very basic and is useful for images that have a slowly changing background and noise. For images with more complicated background and noise statistics it is advised that you use an external program to pre-compute these maps and then feed them into Aegean with the --background and --noise flags. Since I have not come across a program that can calculate these images in a speedy manner I have built one myself.

## Aim 
The quick-and-dirty method for calculating the background and noise of an image is to pass a sliding boxcar filter over the image and, for each pixel, calculate the mean and standard deviation of all pixels within a box centred on that pixel. The problem with this approach is two-fold: one - although it is easy to code it is very time consuming, and two - the standard deviation is biased in the presence of sources. 

The aim of BANE is to provide an accurate measure of the background and noise properties of an image, and to do so in a relatively short amount of time.

## Methodology

There are two main techniques that BANE uses to reduce the compute time for a background and noise calculation, whilst retaining a high level of accuracy.

* Since radio images have a high level of correlation between adjacent pixels, BANE does not calculate the mean and standard deviation for every pixel. It will calculate these quantities on a sparse grid of pixels and then interpolate to give the final background and noise images. For a grid spacing of 5x5 pixels this reduces the total computations by a factor of 25, with only a small amount of time required for interpolation.

* To avoid contamination from source pixels BANE performs sigma clipping. Pixels that are greater than 3sigma from the mean are masked, and this processes is repeated 3 times. The non-masked pixels are then used to calculate the median and std which are equated to be the background and rms.

BANE offers the user a set of parameters that can be used to tune the speed/accuracy to a users desire. The parameters are the grid spacing (in each of the x,y directions), and the size of the box (again in x,y directions) over which the background and noise is calculated. A grid spacing of 1x1 is equivalent to a traditional box-car smooth using the median and std.

Since we define the noise to be the variance about the median, it is necessary for BANE to make two passes over the data: the first pass calculates the background level, and the second pass calculates the deviation from this background level. This requirement doubles the run time of BANE, however for images where the background level is known to be slowly changing (on scales of the box size), a single pass is all that is required.

## Processing steps

The implementation of the process isn't that important but the idea is as follows:

1. select every Nth pixel in the image to form a grid (where N is the grid size, and can be different in the x and y directions).
1. around each grid point draw a box that is MxM pixels wide (where M is the box size, and can be different in the x,y directions).
1. do sigma clipping (3 rounds at 3sigma) to remove the contribution of source pixels
1. calculate the median of all pixels within the box and use that as the background
1. run a linear interpolation between the grid points to make a background image
1. calculate a background subtracted image (data-background)
1. repeat steps 1-4 on the background subtracted image, but instead of calculating the median, use the std.



## Usage

The usage of BANE is described in the help text as follows:
```console
usage: BANE [-h] [--out OUT_BASE] [--grid STEP_SIZE STEP_SIZE] [--box BOX_SIZE BOX_SIZE] [--cores CORES]
            [--stripes STRIPES] [--slice CUBE_INDEX] [--nomask] [--noclobber] [--debug] [--compress] [--cite]
            [image]

positional arguments:
  image

optional arguments:
  -h, --help            show this help message and exit

Configuration Options:
  --out OUT_BASE        Basename for output images default: FileName_{bkg,rms}.fits
  --grid STEP_SIZE STEP_SIZE
                        The [x,y] size of the grid to use. Default = ~4* beam size square.
  --box BOX_SIZE BOX_SIZE
                        The [x,y] size of the box over which the rms/bkg is calculated. Default = 5*grid.
  --cores CORES         Number of cores to use. Default = all available.
  --stripes STRIPES     Number of slices.
  --slice CUBE_INDEX    If the input data is a cube, then this slice will determine the array index of the image which
                        will be processed by BANE
  --nomask              Don't mask the output array [default = mask]
  --noclobber           Don't run if output files already exist. Default is to run+overwrite.
  --debug               debug mode, default=False
  --compress            Produce a compressed output file.
  --cite                Show citation information.
```

## Description of options
* `--compress`: This option causes the output files to be very small. This compression is done by writing a fits image without any interpolation. Files that are produced in this way have extra keys in their fits header, which are recognized by Aegean. When compressed files are loaded by aegean they are interpolated (expanded) to their normal sizes.
* `--nomask`: By default BANE will mask the output image to have the same masked pixels as the input image. This means that nan/blank pixels in the input image will be nan in the output image. This doesn't happen if `--compress` is selected.
* `--stripes`: BANE will break the image into this many sections and process each in turn. By default this is equal to the number of cores, so that all stripes will be processed at the same time. By setting stripes>cores it is possible to reduce the instantaneous memory usage of BANE at the cost of run time.