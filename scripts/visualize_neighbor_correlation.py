

# visualize the correlation between 4-connected neighbors.
#

from datetime import date, datetime
from geo_field import GeoField
from multiprocessing import Pool
from geo_rendering import render_component_single
import matplotlib.pyplot as plt
import numpy as np
import fastcluster
from scipy.cluster.hierarchy import dendrogram, fcluster

#
# Current evaluation params
#
POOL_SIZE = None


if __name__ == "__main__":

    print("Visualization of correlation between neighbors 1.0")
    
    print("Loading data ...")
    gf = GeoField()
    gf.load("data/pres.mon.mean.nc", 'pres')
    gf.transform_to_anomalies()
    gf.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
    gf.slice_spatial(None, [20, 89])
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)
    
    # get the correlation
    num_lats, num_lons = gf.spatial_dims()
    num_gpoints = num_lats * num_lons
    dfC = np.zeros([1, (num_lats - 1) * 2, num_lons], dtype = np.float64)
    
    # compute the neighbor correlations
    y = np.ones(shape=(num_gpoints, num_gpoints)) * 10e6
    Clats = []
    for lat in range(num_lats - 1):
        Clats.append(gf.lats[lat])
        Clats.append(0.5 * (gf.lats[lat] + gf.lats[lat+1]))
        for lon in range(num_lons):
            pt_1 = lat * num_lons + lon
            pt_2 = lat * num_lons + lon + 1
            
            lon1 = (lon + 1) % num_lons
            cc = np.corrcoef(gf.d[:, lat, lon], gf.d[:, lat, lon1], rowvar = 0)[0,1]
            y[pt_1, pt_2] = 1.0 - cc*cc
            dfC[0, 2*lat, lon] = cc

            pt_2 = (lat + 1) * num_lons + lon
            cc = np.corrcoef(gf.d[:, lat, lon], gf.d[:, lat+1, lon], rowvar = 0)[0,1]
            y[pt_1, pt_2] = 1.0 - cc*cc
            dfC[0, 2*lat+1, lon] = cc
            
    ndx = np.triu_indices(num_gpoints, k = 1)
    ytri = y[ndx]
            
    print("Plotting ...")
    gfC = GeoField()
    gfC.d = dfC
    gfC.tm = np.array([0])
    gfC.lons = gf.lons
    gfC.lats = np.array(Clats)
    
    plt.figure()
    plt.hist(dfC.flatten(), bins = 50)
    plt.title('Histogram of neighbor correlations')
    
    plt.figure()
    plt.imshow(dfC[0, :, :])
    plt.colorbar()
    plt.title('Image show of the correlation lattice')
    
    plt.figure()
    plt.imshow(dfC[0, 0::2, :])
    plt.colorbar()
    plt.title('Image show of the correlation lattice - longitudinal')

    plt.figure()
    plt.imshow(dfC[0, 1::2, :])
    plt.colorbar()
    plt.title('Image show of the correlation lattice - lattitudinal')
    
#    plt.figure()
#    plt.imshow(y)
#    plt.title('Distance matrix')

#    render_component_single(gfC.d[0, :, :], gfC.lats, gfC.lons, False, None, "Neighbor correlation")

    print("Clustering ...")
    plt.figure()
    Z = fastcluster.linkage(ytri, method = 'single')
    print("Plotting dendrogram ...")
    dendrogram(Z, 7, 'level')
    
    max_d = np.amax(Z[:,2])
    print("Maximum distance is %g" % max_d)
    my_d = max_d / 2
    cont = True
    while cont:
        f = fcluster(Z, my_d, 'distance')
        print f.shape, my_d
        if np.amax(f) > 30:
            my_d = (max_d + my_d) * 0.5
        elif np.amax(f) < 10:
            my_d = my_d - (max_d - my_d) / max_d
        else:
            cont = False
    
    # now plot the clusters
    f_grid = np.reshape(f, (num_lats, num_lons))
    
    plt.figure()
    plt.imshow(f_grid)
    plt.colorbar()
    plt.title('Clustering of the data')
    
    plt.figure()
    render_component_single(f_grid, gf.lats, gf.lons, False, None, "Cluster assignment")    
    
    plt.show()
