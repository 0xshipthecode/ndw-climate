

import matplotlib
matplotlib.use('Agg')

import geo_rendering as gr
import numpy as np
import sys
import cPickle
import os
import matplotlib.pyplot as plt
from matplotlib import lines
from geo_rendering import plot_data_robinson
import index_data_loader

def load_component(fname, cndx):

    with open(fname, 'r') as f:
        data = cPickle.load(f)

    U = data['Ur'][:, cndx-1]
    ts = np.asarray(data['ts'])[cndx-1,:]

    return U, ts, data['lats'], data['lons']


def standardize(ind):
    print("Standardizing ind, shape = %s mean %g std %g" % (str(ind.shape), np.mean(ind), np.std(ind)))
    return (ind - np.mean(ind)) / np.std(ind)


def chop_series(ts, y1, y2):
    return ts[((y1 - 1948) * 12):((y2 - 1948)*12)]

index_loaders = {
    'SOI' : lambda: index_data_loader.loadPSD('indices/SOI.signal.ascii', 1951, 2012),
    'NINO3.4' : lambda: index_data_loader.loadPSD('indices/nino34.long.data', 1948, 2012),
    'NAO_station' : lambda: index_data_loader.loadPSD('indices/nao_station_monthly_4.txt', 1948, 2012),
    'NAO_pca' : lambda: index_data_loader.loadPSD('indices/nao_pc_monthly_8.txt', 1948, 2012),
    'PNA' : lambda: index_data_loader.loadPSD('indices/pna19482010', 1948, 2010),
    'EA' : lambda: index_data_loader.load_three_column('indices/ea_index.tim', 1950, 2012),
    'WP' : lambda: index_data_loader.load_three_column('indices/wp_index.tim', 1950, 2012),



}

file_names = {
    'SAT' : 'results/sat_all_var_b0_cosweights_varimax_detrended_68.bin',
    'SLP' : 'results/slp_all_var_b0_cosweights_varimax_detrended_60.bin'
}


if __name__ == '__main__':

    if len(sys.argv) != 8:
        print("Usage: publication_figures_triangles.py <index-name> <data_name> <comp_ndx> <data_name> <comp_ndx> <euro_centerd> <output_file>")
        sys.exit(1)

    ind_name = sys.argv[1]
    d1, c1 = sys.argv[2], int(sys.argv[3])
    d2, c2 = sys.argv[4], int(sys.argv[5])
    euro_centered = sys.argv[6].lower() == 'true'
    outf = sys.argv[7]

    U1, ts1, lats1, lons1 = load_component(file_names[d1], c1)
    U2, ts2, lats2, lons2 = load_component(file_names[d2], c2)

    index_data = index_loaders[ind_name]()
    tsi = index_data['ts']

    # years are always whole in the data
    ymin, ymax = min(index_data['years']), max(index_data['years']) + 1
    print("ymin = %d ymax = %d" % (ymin, ymax))
    ts_len = (ymax - ymin) * 12

    print("ts_len %d tsi_len %d" % (ts_len, tsi.shape[0]))

    ts_all = np.zeros((ts_len, 3))
    ts_all[:,0] = standardize(tsi)
    ts_all[:,1] = standardize(chop_series(ts1, ymin, ymax)) - 4.0
    ts_all[:,2] = standardize(chop_series(ts2, ymin, ymax)) - 2*4.0

    cc = np.corrcoef(ts_all, rowvar = 0)

    # all time series data    

    # load data from component sets
    fig = plt.figure(figsize = (12, 6))
    plt.subplots_adjust(left = 0.025, right = 0.975, top = 0.97, wspace = 0.4, hspace = 0.3)
    ax = plt.gca()

    plt.subplot2grid((2,2), (0,0))
    plot_data_robinson(np.reshape(U1, (len(lats1), len(lons1))), lats1, lons1,
                       subplot = True, add_colorbar = False, parallel_labels = 'left',
                       euro_centered = euro_centered)

    plt.subplot2grid((2,2), (0,1))
    plot_data_robinson(np.reshape(U2, (len(lats2), len(lons2))), lats2, lons2,
                       subplot = True, add_colorbar = False, parallel_labels = 'right',
                       euro_centered = euro_centered)

    plt.subplot2grid((2,2), (1,0), colspan = 2)
    plt.plot(ts_all, linewidth = 0.5)
    plt.yticks([])

    year_step = 8
    years = range(1948, 2013, year_step)
    sy = [str(y) for y in years]
    plt.xticks(range(0, ts_all.shape[0], 12 * year_step), sy, rotation = 90)

    # draw the correlations into a triangle
    plt.figtext(0.5, 0.75, ind_name, horizontalalignment = 'center')
    plt.figtext(0.41, 0.53, d1, horizontalalignment = 'center')
    plt.figtext(0.59, 0.53, d2, horizontalalignment = 'center')

    # add triangle
    fig.lines.append(lines.Line2D([0.43, 0.5], [0.535, 0.73], color = 'k',
                     transform = fig.transFigure, figure = fig))
    fig.lines.append(lines.Line2D([0.57, 0.5], [0.535, 0.73], color = 'k',
                     transform = fig.transFigure, figure = fig))
    fig.lines.append(lines.Line2D([0.43, 0.57], [0.535, 0.535], color = 'k',
                     transform = fig.transFigure, figure = fig))


    # add correlations
    plt.figtext(0.445, 0.63, '%.2g' % cc[0, 1], horizontalalignment = 'center')
    plt.figtext(0.56, 0.63, '%.2g' % cc[0, 2], horizontalalignment = 'center')
    plt.figtext(0.5, 0.545, '%.2g' % cc[1, 2], horizontalalignment = 'center')
   
    plt.axis('tight')
    plt.savefig(outf)
    
