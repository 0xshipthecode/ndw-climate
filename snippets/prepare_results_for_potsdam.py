

from datetime import date

import numpy as np
import cPickle
import geo_data_loader
import geo_field


with open('results/sat_all_var_bootstrap_results_b1000_cosweights_varimax.bin') as f:
    d_sat = cPickle.load(f)
    
with open('results/slp_all_var_bootstrap_results_b1000_cosweights_varimax.bin') as f:
    d_slp = cPickle.load(f)
    
# use old loading procedure that includes poles (they were set to zero anyway by the cos multiplication)
gf_slp = geo_field.GeoField()
gf_slp.load('data/slp.mon.mean.nc', 'slp')
gf_slp.transform_to_anomalies()
gf_slp.normalize_monthly_variance()
gf_slp.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
    
slp_data = gf_slp.data()
slp_data *= gf_slp.qea_latitude_weights()
slp_data = np.transpose(np.reshape(slp_data, (slp_data.shape[0], slp_data.shape[1] * slp_data.shape[2])))

sat_comps = d_sat['mean']
sat_comps /= np.sum(sat_comps**2)**0.5

gf_sat = geo_field.GeoField()
gf_sat.load('data/air.mon.mean.nc', 'air')
gf_sat.slice_level(0)
gf_sat.transform_to_anomalies()
gf_sat.normalize_monthly_variance()
gf_sat.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))

sat_data = gf_sat.data()
sat_data *= gf_sat.qea_latitude_weights()
sat_data = np.transpose(np.reshape(sat_data, (sat_data.shape[0], sat_data.shape[1] * sat_data.shape[2])))

slp_comps = d_slp['mean']
slp_comps /= np.sum(slp_comps**2)**0.5

# we extract the time series

sat_comp_ts = np.dot(sat_comps.T, sat_data)
slp_comp_ts = np.dot(slp_comps.T, slp_data)

# store the results

results = { 'sat_comps' : sat_comps, 'sat_comp_ts' : sat_comp_ts,
            'slp_comps' : slp_comps, 'slp_comp_ts' : slp_comp_ts }
with open('results/components_for_potsdam.pickle', 'w') as f:
    cPickle.dump(results, f, 0)
