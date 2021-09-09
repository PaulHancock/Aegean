# AeReg

The regrouping and rescaling operations that were introduced as part of the priorized fitting have been moved into the cluster module.
The script `AeReg` will allow a user to access these operations from the command line such that they can see how the regrouping and rescaling operations will work before having to do the priorized fitting.

```
usage: regroup [-h] --input INPUT --table TABLES [--eps EPS] [--noregroup] [--ratio RATIO] [--psfheader PSFHEADER] [--debug]

optional arguments:
  -h, --help            show this help message and exit

Required:
  --input INPUT         The input catalogue.
  --table TABLES        Table outputs, format inferred from extension.

Clustering options:
  --eps EPS             The grouping parameter epsilon (~arcmin)
  --noregroup           Do not perform regrouping (default False)

Scaling options:
  --ratio RATIO         The ratio of synthesized beam sizes (image psf / input catalog psf).
  --psfheader PSFHEADER
                        A file from which the *target* psf is read.

Other options:
  --debug               Debug mode.
```