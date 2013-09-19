#!/usr/bin/env bash

rm -rf plots/match_*


# slp detrended to sat detrended (space & time)
mkdir plots/match_slp_detrend_to_sat_detrend_space/ plots/match_slp_detrend_to_sat_detrend_time/
python scripts/geo_data_match_components.py results/slp_all_var_b0_cosweights_varimax_detrended.bin results/sat_all_var_b0_cosweights_varimax_detrended.bin space plots/match_slp_detrend_to_sat_detrend_space/ &
python scripts/geo_data_match_components.py results/slp_all_var_b0_cosweights_varimax_detrended.bin results/sat_all_var_b0_cosweights_varimax_detrended.bin time plots/match_slp_detrend_to_sat_detrend_time/ &

# slp to sat (space & time)
mkdir plots/match_slp_to_sat_space/ plots/match_slp_to_sat_time/
python scripts/geo_data_match_components.py results/slp_all_var_b0_cosweights_varimax.bin results/sat_all_var_b0_cosweights_varimax.bin space plots/match_slp_to_sat_space/ &
python scripts/geo_data_match_components.py results/slp_all_var_b0_cosweights_varimax.bin results/sat_all_var_b0_cosweights_varimax.bin time plots/match_slp_to_sat_time/ &

# slp to slp detrend (space & time)
mkdir plots/match_slp_to_slp_detrend_space/ plots/match_slp_to_slp_detrend_time/
python scripts/geo_data_match_components.py results/slp_all_var_b0_cosweights_varimax.bin results/slp_all_var_b0_cosweights_varimax_detrended.bin space plots/match_slp_to_slp_detrend_space/ &
python scripts/geo_data_match_components.py results/slp_all_var_b0_cosweights_varimax.bin results/slp_all_var_b0_cosweights_varimax_detrended.bin time plots/match_slp_to_slp_detrend_time/ &
