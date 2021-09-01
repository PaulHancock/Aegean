# AeRes

If you want to get residual maps, or model maps, from Aegean then this tool is what you are looking for.

AeRes will take an image, and Aegean catalog, and write a new image with all the sources removed. You can also ask for an image that has just the sources in it.

You can use AeRes as shown below:

```
Usage: AeRes -c input.vot -f image.fits -r residual.fits [-m model.fits]

Options:
  -h, --help            show this help message and exit
  -c CATALOG, --catalog=CATALOG
                        Catalog in a format that Aegean understands. RA/DEC
                        should be in degrees, a/b/pa should be in
                        arcsec/arcsec/degrees.
  -f FITSFILE, --fitsimage=FITSFILE
                        Input fits file.
  -r RFILE, --residual=RFILE
                        Output residual fits file.
  -m MFILE, --model=MFILE
                        Output model file [optional].
  --add                 Add components instead of subtracting them.
  --mask                Instead of subtracting sources, just mask them
  --sigma=SIGMA         If masking, pixels above this SNR are masked (requires
                        input catalogue to list rms)
  --frac=FRAC           If masking, pixels above frac*peak_flux are masked for
                        each source
  --debug               Debug mode.
```

The acceptable formats for the catalogue file are anything that Aegean can write. Use `aegean.py --tformats` to see the formats that Aegean can support on your machine. Usually the best idea is to just edit a table that Aegean has created.