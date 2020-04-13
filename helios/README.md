# HELIOS

The `helios` module of the `PlasmaML` repository contains utilities for 
training tensorflow based models on solar data sets.

This repo is also the home of the reference implementation of the [_Dynamic Time Lag Regression_](https://openreview.net/forum?id=SkxybANtDB) framework and its application on the solar wind prediction task.


## DTLR Solar wind prediction.

Run the DTLR cross validation experiment with

```
./helios/scripts/csss_cv_runner.sh <experiment id>
```

After the DTLR results are stored in the folder `<experiment id>`, run the fixed lag baseline model.

```
./helios/scripts/csss_cv_bs_runner.sh <top dir> <experiment id>
```

