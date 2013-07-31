#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:src:lib
scripts/plot_components.py results/sat_all_var_b0_cosweights_varimax.bin plots/sat_all_pacific sat_all false &
scripts/plot_components.py results/sat_all_var_b0_cosweights_varimax_detrended.bin plots/sat_all_detrend_pacific sat_all_detrend false &
scripts/plot_components.py results/slp_all_var_b0_cosweights_varimax.bin plots/slp_all_pacific slp_all false &
scripts/plot_components.py results/slp_all_var_b0_cosweights_varimax_detrended.bin plots/slp_all_detrend_pacific slp_all_detrend false &
