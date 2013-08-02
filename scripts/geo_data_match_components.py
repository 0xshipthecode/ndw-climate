
import matplotlib
matplotlib.use('agg')

from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from component_analysis import pca_components_gf, orthomax, match_components_from_matrix
from geo_rendering import plot_data_robinson

import math
import os.path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scist
import cPickle
from datetime import date, datetime
import time




if __name__ == '__main__':

    if len(sys.argv) < 5:
        print("Usage: geo_data_match_components.py <dataset1> <dataset2> <time|space> <output-dir>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        d1 = cPickle.load(f)

    with open(sys.argv[2], 'r') as f:
        d2 = cPickle.load(f)

    out_dir = sys.argv[4]

    # retrieve data from results
    lats1, lons1 = d1['lats'], d1['lons']
    Nlats1, Nlons1 = len(lats1), len(lons1)
    lats2, lons2 = d2['lats'], d2['lons']
    Nlats2, Nlons2 = len(lats2), len(lons2)
    comps1, comps2 = d1['Ur'], d2['Ur']
    ts1, ts2 = d1['ts'], d2['ts']

    # find matching of components by time or by space
    C, OC, sign_flip, perm = None, None, None, None
    if sys.argv[3] == 'time':
        m1 = np.transpose(np.asarray(ts1))
        m2 = np.transpose(np.asarray(ts2))
        N1, N2 = m1.shape[1], m2.shape[1]

        C = np.corrcoef(m1, m2, rowvar = 0)
        C = C[0:N1, N1:N1+N2]
        perm, sign_flip = match_components_from_matrix(C)
        sign_flip = sign_flip[0,:]

        om1 = np.copy(np.asarray(comps1))
        om1 /= np.sum(om1**2, axis = 0)
        om2 = np.copy(np.asarray(comps2))
        om2 /= np.sum(om2**2, axis = 0)

        if om1.shape[0] == om2.shape[0]:
            OC = np.dot(np.transpose(om1), om2)
        else:
            OC = -10 * np.ones((om1.shape[1], om2.shape[1]))

    elif sys.argv[3] == 'space':
        m1 = np.copy(np.asarray(comps1))
        m1 /= np.sum(m1**2, axis = 0)
        m2 = np.copy(np.asarray(comps2))
        m2 /= np.sum(m2**2, axis = 0)

        C = np.dot(np.transpose(m1), m2)
        perm, sign_flip = match_components_from_matrix(C)
        sign_flip = sign_flip[0,:]

        om1 = np.transpose(np.asarray(ts1))
        om2 = np.transpose(np.asarray(ts2))
        N1, N2 = om1.shape[1], om2.shape[1]
        OC = np.corrcoef(om1, om2, rowvar = 0)
        OC = OC[0:N1, N1:N1+N2]

    else:
        print("Invalid method, must be either 'time' or 'space'.")

    # standardize time series
    ts1 /= np.var(ts1, axis = 1)
    ts2 /= np.var(ts2, axis = 1)

    with open(os.path.join(out_dir, 'matching'), 'w') as f:
        f.write("# matching from %s (left) to %s (right) in %s\n" % (sys.argv[1], sys.argv[2], sys.argv[3]))
        f.write("# third column - match similarity (correlation in time, dot product in space)\n")
        f.write("# fourth column - complementary similarity (time -> dot product in space, space -> correlation in time\n")
        for i, p in zip(range(C.shape[0]), perm):
            f.write("%d, %d, %g, %g\n" % (i+1, p+1, C[i,perm[i]], OC[i,perm[i]]))
    
    print("Rendering images")
    for i in range(len(perm)):
        if perm[i] >= 0:
            f = plt.figure(figsize = (12, 6))
            gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[5, 5],height_ratios=[6,4]) 
            plt.subplot(gs[0,0])
            plot_data_robinson(np.reshape(comps1[:, i], (Nlats1, Nlons1)),
                               lats1, lons1, subplot = True, euro_centered = True) 
            plt.title('Component %d' % (i+1))
            plt.subplot(gs[0,1])
            plot_data_robinson(np.reshape(comps2[:, perm[i]] * sign_flip[i], (Nlats2, Nlons2)),
                               lats2, lons2, subplot = True, euro_centered = True)
            plt.title('Component %d%s' % (perm[i]+1, " (sign fliped)" if sign_flip[i] == -1 else ""))
            plt.subplot(gs[1,:])
            plt.plot(ts1[i,:].T, 'b-')
            plt.plot(ts2[perm[i],:].T * sign_flip[i], 'g-')
            if sys.argv[3] == 'time':
                plt.figtext(0.25, 0.43, 'match by timeseries Pearson corr %g, components dot %g' % (C[i,perm[i]], OC[i,perm[i]]))
            else:
                plt.figtext(0.25, 0.43, 'match by component dot %g, time series Pearson corr %g' % (C[i,perm[i]], OC[i,perm[i]]))
            plt.savefig(os.path.join(out_dir, "match_%d_to_%d.png" % (i+1, perm[i]+1)))
            plt.clf()
            sys.stdout.write("*")

    print("\nDone.")
        
