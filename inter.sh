#!/usr/bin/env bash

PYTHONPATH=$PYTHONPATH:src:lib
export PYTHONPATH
ipython -i -c 'from inter_utils import *; import numpy as np'

