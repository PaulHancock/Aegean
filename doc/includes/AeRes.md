# AeRes

If you want to get residual maps, or model maps, from Aegean then this tool is what you are looking for.

AeRes will take an image, and Aegean catalog, and write a new image with all the sources removed. You can also ask for an image that has just the sources in it.

You can use AeRes as shown below:

```console
usage: AeRes [-h] [-c CATALOG] [-f FITSFILE] [-r RFILE] [-m MFILE] [--add] [--mask] [--sigma SIGMA] [--frac FRAC]
             [--racol RA_COL] [--deccol DEC_COL] [--peakcol PEAK_COL] [--acol A_COL] [--bcol B_COL] [--pacol PA_COL]
             [--debug]

optional arguments:
  -h, --help            show this help message and exit

I/O arguments:
  -c CATALOG, --catalog CATALOG
                        Catalog in a format that Aegean understands. RA/DEC should be in degrees, a/b/pa should be in
                        arcsec/arcsec/degrees.
  -f FITSFILE, --fitsimage FITSFILE
                        Input fits file.
  -r RFILE, --residual RFILE
                        Output residual fits file.
  -m MFILE, --model MFILE
                        Output model file [optional].

Config options:
  --add                 Add components instead of subtracting them.
  --mask                Instead of subtracting sources, just mask them
  --sigma SIGMA         If masking, pixels above this SNR are masked(requires input catalogue to list rms)
  --frac FRAC           If masking, pixels above frac*peak_flux are masked for each source

Catalogue options:
  --racol RA_COL        RA column name
  --deccol DEC_COL      Dec column name
  --peakcol PEAK_COL    Peak flux column name
  --acol A_COL          Major axis column name
  --bcol B_COL          Minor axis column name
  --pacol PA_COL        Position angle column name

Extra options:
  --debug               Debug mode.
```

The acceptable formats for the catalogue file are anything that Aegean can write. Use `aegean.py --tformats` to see the formats that Aegean can support on your machine. Usually the best idea is to just edit a table that Aegean has created.