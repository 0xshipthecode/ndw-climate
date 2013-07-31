#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import geo_rendering as gr
import numpy as np
import sys
import cPickle
import os
import matplotlib.pyplot as plt


def tofield(d, lats, lons):
    return d.reshape([len(lats), len(lons)])


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: plot_components.py <component_file> <output_directory> <prefix> [euro_centered?]")
        sys.exit(1)

    # by default euro_centered is set to True
    if len(sys.argv) < 5:
        euro_centered = True
    else:
        euro_centered = (sys.argv[4].lower() == "true")

    # read in args
    comp_file = sys.argv[1]
    output_dir = sys.argv[2]
    prefix = sys.argv[3]

    with open(comp_file, 'r') as f:
        data = cPickle.load(f)

    # create output directory if non-existent
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load results & their locations
    lats, lons = data['lats'], data['lons']
    cmps = data['Ur']
    ts = np.asarray(data['ts'])
    expvar = data['expvar']
    regexpvar = data['reg_expvar']
    total_var = data['total_var']
    s2 = data['s2']

    # mark binary components using a simplified method (not clustering compatible)
    bin_cmps = cmps.argmax(1)

    # this will plot the components in descending order of variance
    # (the result file is already sorted this way)
    centers = []
    for i in range(cmps.shape[1]):
        c_i = tofield(cmps[:,i], lats, lons)
        flat_index = np.argmax(c_i)
        clat, clon = np.unravel_index(flat_index, c_i.shape)
        center_i = (lons[clon], lats[clat])
        centers.append(center_i)
        gr.plot_component_robinson(c_i, ts[i,:], lats, lons, center_i, regexpvar[i] / total_var,
                                   euro_centered = euro_centered,
                                   filename = os.path.join(output_dir, '%s_comp_%d.png' % (prefix, i+1)))
        plt.close("all")

    # this will plot the clusters on one image
    filename = os.path.join(output_dir, '%s_components.png' % prefix)
    gr.plot_clusters_robinson(tofield(bin_cmps, lats, lons), lats, lons, centers = centers,
                              filename = filename)
