

from datetime import date
from geo_field import GeoField
from geo_rendering import render_component_single

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
#    gf.slice_spatial(None, [20, 89])
    gf.slice_date_range(date(1950, 1, 1), date(2012, 3, 1))
    
    # unroll the data
    ts = gf.data()
    
    # load the monthly NAO index
    nao = np.loadtxt('data/nao_index.tim.txt', skiprows = 0)

    naoh = np.loadtxt('data/nao_index_hurrel.tim.txt', skiprows = 0)
    naoh_ndx = naoh[:, 2]

    nao_ndx = nao[:naoh_ndx.shape[0], 2]

    print nao_ndx.shape
    print naoh_ndx.shape
    ts = ts[:naoh_ndx.shape[0], :, :]
    
    #compute the correlation map to NAO
    C = np.zeros_like(gf.d[0, :, :])
    Ch = np.zeros_like(gf.d[0, :, :])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i, j] = np.corrcoef(nao_ndx, ts[:, i, j], rowvar = 0)[0,1]
            Ch[i, j] = np.corrcoef(naoh_ndx, ts[:, i, j], rowvar = 0)[0,1]
            
    # plot using basemap
#    render_component_single(C, gf.lats, gf.lons, False, None, 'NAO index correlation')
    
    # plot using basemap
#    render_component_single(Ch, gf.lats, gf.lons, False, None, 'NAO index correlation - Hurrel')

    # compute the variance of the data and plot it
#    render_component_single(np.var(ts, axis = 0), gf.lats, gf.lons, False, None, 'Variance of Data')
    
    plt.figure()
    plt.plot(nao_ndx, 'r-')
    plt.plot(naoh_ndx, 'b-')
    plt.show()
    
    print(np.corrcoef(nao_ndx, naoh_ndx, rowvar = False))
