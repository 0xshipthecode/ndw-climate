


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
NUM_SURR = 400
NUM_EIGS = 20
POOL_SIZE = None
RECOMPUTE_MODEL = True


def compute_surrogate_cov_eigvals(U):

    # U are the PCA components
    rvdims = np.zeros((NUM_EIGS,))
    
    Ri = U.copy()
    # permute the values in each column of Ri
    for i in range(NUM_EIGS):
        Ri[:, i] = np.random.permutation(Ri[:,i])
    
    for i in range(NUM_EIGS):
        
        # compute the eigenvalues of the covariance matrix
        rlam = pca_eigvals(Ri[:, i:])
        
        # compute RVDIM(1)
        rvdims[i] = rlam[0] / np.sum(rlam**2)**0.5
    
    return rvdims


if __name__ == "__main__":

    print("Estimate PCA components script version 1.0")
    
    S = np.zeros(shape = (20, 50), dtype = np.int32)
    S[10:18, 25:45] = 1
    S[0:3, 6:12] = 2
    S[8:15, 2:12] = 3
    v, Sr = constructVAR(S, [0.0, 0.6, 0.9, 0.7], [0.3, 0.5], [0.0, 0.0])
    ts = v.simulate(200)
    gf = make_model_geofield(S, ts)
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)
    
    # compute the eigenvalues/eigenvectos of the covariance matrix of
    Ud, dlam, _ = pca_components_gf(gf.data())
    drdims = np.zeros((NUM_EIGS,))
    for i in range(NUM_EIGS):
        drdims[i] = dlam[i] / np.sum(dlam[i:]**2)**0.5
    
    sd = SurrGeoFieldAR([0, 30], 'sbc')
    sd.copy_field(gf)
    sd.prepare_surrogates(pool)
    srdims = np.zeros((NUM_SURR, NUM_EIGS))
    
    # generate and compute eigenvalues for 20000 surrogates
    t1 = datetime.now()
    
    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    print("Running parallel generation of surrogates and SVD")
    slam_list = pool.map(compute_surrogate_cov_eigvals, [Ud] * NUM_SURR)
    
    # rearrange into numpy array (can I use vstack for this?)
    for i in range(len(slam_list)):
        srdims[i, :] = slam_list[i]
        
    srdims.sort(axis = 0)
        
    print("Saving computed spectra ...")
    
    plt.figure()
    plt.plot(np.arange(NUM_EIGS) + 1, drdims, 'ro-')
    plt.errorbar(np.arange(NUM_EIGS) + 1, np.mean(srdims, axis = 0), np.std(srdims, axis = 0) * 3, fmt = 'g-')
    plt.plot(np.arange(NUM_EIGS) + 1, srdims[int(NUM_SURR * 0.95),:], 'kx-', linewidth = 2)

    plt.show()
    print("DONE.")
