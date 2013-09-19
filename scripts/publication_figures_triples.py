

import matplotlib
matplotlib.use('Agg')

import geo_rendering as gr
import numpy as np
import sys
import cPickle
import os
import matplotlib.pyplot as plt
from geo_rendering import plot_data_robinson
from geo_data_loader import load_monthly_sat_all

def load_component(fname, cndx):

    with open(fname, 'r') as f:
        data = cPickle.load(f)

    U = data['Ur'][:, cndx-1]
    ts = np.asarray(data['ts'])[cndx-1,:]

    return U, ts, data['lats'], data['lons']


def load_data(fname):
    with open(fname, 'r') as f:
        data = cPickle.load(f)
    return data



if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: pub_figures.py <slp_ndx> <euro_centered>")
        sys.exit(1)

    slp_ndx = int(sys.argv[1])
    euro_centered = sys.argv[2].lower() == 'true'

    # load data from component sets
    U1, ts1, lats1, lons1 = load_component('results/slp_all_var_b0_cosweights_varimax_detrended_60.bin', slp_ndx)
    data2 = load_data('results/sat_all_var_b0_cosweights_varimax_detrended_68.bin')
    ts2 = np.asarray(data2['ts'])

    # find best matching component in time
    best_match = np.argmax([np.abs(np.corrcoef(ts1, ts2[i,:])[0,1]) for i in range(ts2.shape[0])])

    ts2 = ts2[best_match, :]
    U2 = data2['Ur'][:, best_match]
    lats2, lons2 = data2['lats'], data2['lons']

    gf = load_monthly_sat_all()
    gf.detrend()
    d = gf.data()
    d = np.reshape(d, (768, 71*144))

    # compute correlation of slp time series with sat index
    Uc = np.zeros(U1.shape)
    for i in range(Uc.shape[0]):
        Uc[i] = np.corrcoef(ts1, d[:,i])[0,1]

    plt.figure(figsize = (6, 3 * 4))
    plt.subplots_adjust(left = 0.02, right = 0.96, hspace = 0.5)

    plt.subplot(311)
    plot_data_robinson(np.reshape(U1, (len(lats1), len(lons1))), lats1, lons1,
                       subplot = True, euro_centered = euro_centered)

    plt.subplot(312)
    plot_data_robinson(np.reshape(U2, (len(lats2), len(lons2))), lats2, lons2,
                       subplot = True, euro_centered = euro_centered)

    plt.subplot(313)
    plot_data_robinson(np.reshape(Uc, (len(lats2), len(lons2))), lats2, lons2,
                       subplot = True, euro_centered = euro_centered)

    plt.savefig('slp_%02d_sat_%02d.eps' % (slp_ndx, best_match+1))
    
        
        
    
        
    


