# SR6

[BANE](./BANE) is able to output compressed background and rms images using the `--compress` option. If you have a compressed file and want to expand it to have the same number of pixels as your original image then you need to use SR6.

If you have an image that you, for some reason, want to compress using a super-lossy algorithm known as decimation, then SR6 is what you want.

Usage is:

```
usage: SR6.py [-h] [-o OutputFile] [-f factor] [-x]
              [-i {linear,nearest,cubic}] [-m MaskFile] [--debug] [--version]
              infile

optional arguments:
  -h, --help            show this help message and exit

Shrinking and expanding files:
  infile                input filename
  -o OutputFile         output filename
  -f factor             reduction factor
  -x                    Operation is expand instead of compress.
  -i {linear,nearest,cubic}
                        Interpolation method
  -m MaskFile           File to use for masking pixels.
  --debug               Debug output
  --version             show program's version number and exit
```

In order to be able to expand a file, the file needs to have some special keywords in the fits header. These are inserted automatically by BANE, but you could probably fidget them for yourself if you had the need.

You should be able to shrink any file that you choose.