

# visualize the correlation between 4-connected neighbors.
#

from datetime import date, datetime
from geo_field import GeoField
from multiprocessing import Pool
from geo_rendering import render_component_single
import matplotlib.pyplot as plt
import numpy as np


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
#    gf.slice_spatial(None, [20, 89])
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)
    
    # get the correlation
    num_lats, num_lons = gf.spatial_dims()
    dfC = np.zeros([1, (num_lats - 1) * 2, num_lons], dtype = np.float64)
    
    # compute the neighbor correlations
    Clats = []
    for lat in range(num_lats - 1):
        Clats.append(gf.lats[lat])
        Clats.append(0.5 * (gf.lats[lat] + gf.lats[lat+1]))
        for lon in range(num_lons):
            lon1 = (lon + 1) % num_lons
            dfC[0, 2*lat, lon] = np.corrcoef(gf.d[:, lat, lon], gf.d[:, lat, lon1], rowvar = 0)[0,1] 
            dfC[0, 2*lat+1, lon] = np.corrcoef(gf.d[:, lat, lon], gf.d[:, lat+1, lon], rowvar = 0)[0,1]
            
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

    render_component_single(gfC.d[0, :, :], gfC.lats, gfC.lons, False, None, "Neighbor correlation")
    plt.show()
    