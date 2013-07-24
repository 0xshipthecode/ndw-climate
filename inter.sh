#!/usr/bin/env bash

PYTHONPATH=$PYTHONPATH:src:lib
export PYTHONPATH
ipython -i -c 'from inter_utils import *; import numpy as np; from geo_data_loader import load_monthly_slp_all, load_monthly_sat_all'

