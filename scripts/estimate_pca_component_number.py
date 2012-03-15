


from datetime import date, datetime
from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from multiprocessing import Pool
from component_analysis import pca_eigvals
from spatial_model_generator import constructVAR, make_model_geofield


import os.path
import numpy as np
import matplotlib.pylab as plt
import cPickle

#
# Current simulation parameters
#
NUM_SURR = 500
NUM_EIGS = 50
POOL_SIZE = None
RECOMPUTE_MODEL = True


def compute_surrogate_cov_eigvals(sd):
    sd.construct_surrogate_with_noise()
    return pca_eigvals(sd.surr_data())[:NUM_EIGS]


if __name__ == "__main__":

    print("Estimate PCA components script version 1.0")
    
    print("Loading data ...")
#    d = GeoField()
#    d.load("data/pres.mon.mean.nc", 'pres')
#    d.transform_to_anomalies()
#    d.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
#    d.slice_spatial(None, [20, 89])

    S = np.zeros(shape = (5, 10), dtype = np.int32)
    S[1:4, 0:2] = 1
    S[0:3, 6:9] = 2
    v, Sr = constructVAR(S, [0.0, 0.191, 0.120], [-0.1, 0.1], [0.00, 0.00], [0.01, 0.01])
    ts = v.simulate(768)
    d = make_model_geofield(S, ts)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.corrcoef(ts, rowvar = 0), interpolation = 'nearest')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(S, interpolation = 'nearest')
    plt.colorbar()

#    with open('data/test_gf.bin', 'r') as f:
#        d = cPickle.load(f)
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)
    
    # compute the eigenvalues of the covariance matrix of
    dlam = pca_eigvals(d.data())[:NUM_EIGS]
    dlam = dlam
    
    sd = SurrGeoFieldAR([0, 30], 'sbc')
#    if os.path.exists('data/saved_slp_surrogate_field.bin') and not RECOMPUTE_MODEL:
#        print("Loading existing surrogate model ...")
#        sd.load_field('data/saved_slp_surrogate_field.bin')
#    else:
#        print("Rerunning preparation of surrogates ...")
    sd.copy_field(d)
    sd.prepare_surrogates(pool)
#        print("Saving surrogate models to file (max order: %d) ..." % sd.max_ord)
#        sd.save_field('data/saved_slp_surrogate_field.bin')
    
    slam = np.zeros((NUM_SURR, NUM_EIGS))
    
    # generate and compute eigenvalues for 20000 surrogates
    t1 = datetime.now()
    
    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    print("Running parallel generation of surrogates and SVD")
    slam_list = pool.map(compute_surrogate_cov_eigvals, [sd] * NUM_SURR)
    
    # rearrange into numpy array (can I use vstack for this?)
    for i in range(len(slam_list)):
        slam[i, :] = slam_list[i]
        
    print("Saving computed spectra ...")
                
    # save the results to file
    with open('data/slp_eigvals_surrogates.bin', 'w') as f:
        cPickle.dump([dlam, slam, sd.model_orders(), sd.lons, sd.lats], f)
    
    plt.figure()
    plt.plot(np.arange(NUM_EIGS) + 1, dlam, 'ro-')
    plt.errorbar(np.arange(NUM_EIGS) + 1, np.mean(slam, axis = 0), np.std(slam, axis = 0), fmt = 'g-')
    plt.show()
    
    print("DONE.")
