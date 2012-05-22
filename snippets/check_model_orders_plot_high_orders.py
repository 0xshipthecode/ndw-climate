

import numpy as np
import matplotlib.pyplot as plt
import geo_data_loader
from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
import multiprocessing
import geo_rendering

def find_and_reset(mot):
    i, j = np.nonzero(mot)
    
    if len(i) == 0:
        return None
    
    # get first nonzero point
    i1, j1, mo1 = i[0], j[0], mot[i[0], j[0]]
    
    # find all connected points and among them, select the one with the highest model order
    q = [(i1, j1)]
    while len(q) > 0:
        ci, cj = q.pop()
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            cdi, cdj = ci+di,cj+dj
            if cdi >= 0 and cdj >= 0 and cdi < mot.shape[0] and cdj < mot.shape[1]:
                if mot[cdi, cdj] > 0:
                    q.append((cdi, cdj))
                    if mot[cdi, cdj] > mo1:
                        i1, j1, mo1 = cdi, cdj, mot[cdi, cdj]
                    mot[cdi, cdj] = 0
    
    return i1, j1, mo1
    

gf = geo_data_loader.load_monthly_slp_all()
sgf = SurrGeoFieldAR()
sgf.copy_field(gf)

pool = multiprocessing.Pool(3)
sgf.prepare_surrogates(pool)
pool.close()

# plot the model orders
geo_rendering.render_component_single(sgf.model_orders(), gf.lats, gf.lons, None, None, 'Model orders for SLP')

# look at time series of places with highest model orders (select one point from each connected region)
mo = sgf.model_orders()
mot = mo * (mo > 9)

loc = find_and_reset(mot) 

while loc:
    i, j, m = loc
    plt.figure()
    plt.plot(gf.d[:, i, j])
    plt.title('Position [%g lat, %g lon], model order %d' % (gf.lats[i], gf.lons[j], m))
    loc = find_and_reset(mot)
    