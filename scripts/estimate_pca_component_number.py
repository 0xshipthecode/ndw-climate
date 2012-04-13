


from datetime import date, datetime
from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from multiprocessing import Pool
from component_analysis import pca_eigvals, pca_components, pca_components_gf,\
    match_components_munkres
from spatial_model_generator import constructVAR, make_model_geofield


import os.path
import numpy as np
import matplotlib.pylab as plt
import cPickle

#
# Current simulation parameters
#
NUM_SURR = 1000
NUM_EIGS = 100
POOL_SIZE = None
RECOMPUTE_MODEL = True
COSINE_REWEIGHTING = True

def compute_surrogate_cov_eigvals(x):
    sd, U = x
#    sd.construct_surrogate_with_noise()
    sd.construct_white_noise_surrogates()
#    sd.construct_fourier1_surrogates()
    
    d = sd.surr_data()
    if COSINE_REWEIGHTING:
        d = d * sd.qea_latitude_weights()
    
    Ur, sr, _ = pca_components_gf(d)
    
#    perm, sf = match_components_munkres(U, Ur)
#    Ur = Ur[:, perm[:NUM_EIGS]]
#    Ur *= sf
    
#    return sr[perm[:NUM_EIGS]]
    return sr[:NUM_EIGS], np.amax(np.abs(Ur[:, :NUM_EIGS]), axis = 0)


if __name__ == "__main__":

    print("Estimate PCA components script version 1.0")
    
    print("Loading data ...")
    gf = GeoField()
    gf.load("data/pres.mon.mean.nc", 'pres')
    gf.transform_to_anomalies()
    gf.normalize_monthly_variance()
    gf.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
    gf.slice_spatial(None, [20, 89])
#    gf.slice_spatial(None, [-88, 88])

#    S = np.zeros(shape = (5, 10), dtype = np.int32)
#    S[1:4, 0:2] = 1
#    S[0:3, 6:9] = 2
#    v, Sr = constructVAR(S, [0.0, 0.7, 0.6], [-0.3, 0.3], [0.2, 0.27])
#    ts = v.simulate(768)
#    d = make_model_geofield(S, ts)
    
#    S = np.zeros(shape = (20, 50), dtype = np.int32)
#    S[10:18, 25:45] = 1
#    S[0:3, 6:12] = 2
#    S[8:15, 2:12] = 3
#    v, Sr = constructVAR(S, [0.0, 0.6, 0.9, 0.7], [0.3, 0.5], [0.0, 0.0])
#    ts = v.simulate(200)
#    gf = make_model_geofield(S, ts)

#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.imshow(np.corrcoef(ts, rowvar = 0), interpolation = 'nearest')
#    plt.colorbar()
#    plt.subplot(1,2,2)
#    plt.imshow(S, interpolation = 'nearest')
#    plt.colorbar()

#    with open('data/test_gf.bin', 'r') as f:
#        d = cPickle.load(f)
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)
    
    # compute the eigenvalues/eigenvectos of the covariance matrix of
    d = gf.data()
    if COSINE_REWEIGHTING:
        d = d * gf.qea_latitude_weights()
        
    Ud, dlam, _ = pca_components_gf(d)
    Ud = Ud[:, :NUM_EIGS]
    dlam = dlam[:NUM_EIGS]
    
    sd = SurrGeoFieldAR([0, 30], 'sbc')
    sd.copy_field(gf)
    sd.prepare_surrogates(pool)
    slam = np.zeros((NUM_SURR, NUM_EIGS))
    maxU = np.zeros((NUM_SURR, NUM_EIGS))
    
    # generate and compute eigenvalues for 20000 surrogates
    t1 = datetime.now()
    
    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    print("Running parallel generation of surrogates and SVD")
    slam_list = pool.map(compute_surrogate_cov_eigvals, [(sd, Ud)] * NUM_SURR)
    
    # rearrange into numpy array (can I use vstack for this?)
    for i in range(len(slam_list)):
        slam[i, :], maxU[i, :] = slam_list[i]
        
    maxU.sort(axis = 0)
        
    print("Saving computed spectra ...")
                
    # save the results to file
    with open('data/slp_eigvals_surrogates.bin', 'w') as f:
        cPickle.dump([dlam, slam, sd.model_orders(), sd.lons, sd.lats], f)
    
    plt.figure()
    plt.plot(np.arange(NUM_EIGS) + 1, dlam, 'ro-')
    plt.errorbar(np.arange(NUM_EIGS) + 1, np.mean(slam, axis = 0), np.std(slam, axis = 0) * 3, fmt = 'g-')
    
    plt.figure()
    plt.errorbar(np.arange(NUM_EIGS) + 1, np.mean(maxU, axis = 0), np.std(maxU, axis = 0) * 3, fmt = 'g-')
    plt.plot(np.arange(NUM_EIGS) + 1, np.amax(maxU, axis = 0), 'r-')
    plt.plot(np.arange(NUM_EIGS) + 1, np.amin(maxU, axis = 0), 'r-')
    plt.plot(np.arange(NUM_EIGS) + 1, maxU[94, :], 'bo-', linewidth = 2)
    plt.plot(np.arange(NUM_EIGS) + 1, np.amax(np.abs(Ud), axis = 0), 'kx-', linewidth = 2)

    plt.show()
    print("DONE.")
