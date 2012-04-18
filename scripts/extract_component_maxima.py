


from datetime import date
from geo_field import GeoField
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import scipy.io as sio

if __name__ == '__main__':
    

    # load geo-field
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.transform_to_anomalies()
    gf.normalize_monthly_variance()
    gf.slice_spatial(None, [20, 89])
    gf.slice_date_range(date(1950, 1, 1), date(2012, 3, 1))
    
    # load the components
    with open('results/slp_nh_var_bootstrap_results_b1000_cosweights.bin', 'r') as f:
        d = cPickle.load(f)
        
    # convert to unit vectors
    mn = d['mean']
    mn = mn / np.sum(mn**2, axis = 0) ** 0.5
    
    # mark maxima
    mx_pos = np.argmax(mn**2, axis = 0)
    print mx_pos
    
    mx_loc = np.zeros_like(mn)
    mx_loc[(mx_pos, np.arange(mn.shape[1]))] = 1.0
    
    mx_loc_gf = gf.reshape_flat_field(mx_loc)
    
    sio.savemat("results/component_maxima_locations.mat", { 'mx_loc' : mx_loc_gf })
