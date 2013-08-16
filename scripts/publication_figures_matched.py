

import matplotlib
matplotlib.use('Agg')

import geo_rendering as gr
import numpy as np
import sys
import cPickle
import os
import matplotlib.pyplot as plt
from geo_rendering import plot_data_robinson

def subsample_component_2x2(comp, Nlats, Nlons):
    c2 = np.reshape(comp, (Nlats, Nlons))
    c2 = c2[0::2, ::2]
    return np.reshape(c2, (np.prod(c2.shape),))


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: pub_figures.py <output_file> <component-file> <comp-ndx>  [<component-file>]")
        sys.exit(1)

    output_file = sys.argv[1]
    euro_centered = sys.argv[2].lower() == 'true'
    comp_ndx = int(sys.argv[3])
    Ncomps = len(sys.argv) - 4

    # load data from component sets
    plt.figure(figsize = (6, 2.5*(Ncomps+1)))
    plt.subplots_adjust(top = 0.99, bottom = 0.05)

    time_series = None
    for i in range(Ncomps):
        with open(sys.argv[i+4], 'r') as f:
            data = cPickle.load(f)

        if i == 0:
            # select first component using comp_ndx
            comp_orig = comp = data['Ur'][:, comp_ndx]
            ts_orig = ts = np.asarray(data['ts'])[comp_ndx, :]
            time_series = np.zeros((len(ts), Ncomps))
            time_series[:, i] = ts / np.var(ts) - i*0.4
            lats, lons = data['lats'], data['lons']
            Nlats, Nlons = len(lats), len(lons)
            olats, olons, oNlats, oNlons = lats, lons, Nlats, Nlons
        else:
            # select spatially best matching component if size is the same, else
            # temporally best matching component
            lats, lons = data['lats'], data['lons']
            Nlats, Nlons = len(lats), len(lons)
            tsi = np.asarray(data['ts'])
            compsi = data['Ur']
            if Nlats == oNlats and Nlons == oNlons:
                # we can match by space
                dots = np.dot(comp_orig, compsi)
                comp_ndx = np.argmax(np.abs(dots))
                sign_flip = -1.0 if dots[comp_ndx] < 0 else 1.0
            else:
                # we must match by time
                coss = subsample_component_2x2(comp_orig, oNlats, oNlons)
                dots = np.dot(coss, compsi)
                comp_ndx = np.argmax(np.abs(dots))
                sign_flip = -1.0 if dots[comp_ndx] < 0 else 1.0

            time_series[:, i] = tsi[comp_ndx, :] / np.var(tsi[comp_ndx, :]) * sign_flip - i*0.4
            comp = compsi[:, comp_ndx] * sign_flip

        print("i = %d comp_ndx = %d\n" % (i, comp_ndx))

        plt.subplot(Ncomps+1, 1, i+1)
        plot_data_robinson(np.reshape(comp, (Nlats, Nlons)), lats, lons,
                           subplot = True, euro_centered = euro_centered)
        plt.text(0.0, 0.99*plt.ylim()[1], '(%s)' % ("abcdefgh"[i]))

    plt.subplot(Ncomps+1, 1, Ncomps+1)
    plt.plot(time_series, linewidth = 0.5)
    plt.yticks([])

    year_step = 8
    years = range(1948,2013,year_step)
    sy = [str(y) for y in years]
    plt.xticks(range(0, time_series.shape[0], 12*year_step),sy,rotation = 90)
    
    plt.axis('tight')
    plt.savefig(output_file)
    
        
        
    
        
    


