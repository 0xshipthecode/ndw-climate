ndw-climate
===========

ndw-climate is a python code for detecting components in climatic datasets.
Loaders are provided for reading in netcdf files containing monthly and daily
NCAR/NCEP reanalysis data (Kistler et al. 2001, Kalnay et al. 1996).


Pre-requisites
=============

* python 2.6 or 2.7
* [netCDF4](https://code.google.com/p/netcdf4-python/ "netCDF4")
* numpy (tested with version 1.6.1)
* scipy (tested with version 0.10.1)
* matplotlib (tested with version 1.1.1.rc)
* basemap (tested with version 1.0.2)


Running the code
================

The main computationally-intensive part is estimating the number of components in the dataset.
This is accomplished by editing the script ''scripts/geo_data_estimate_component_count.py''
and running it on a sufficiently strong machine (the number of concurrent workers can be set
in the file).


