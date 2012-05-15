

from datetime import date, datetime
from geo_field import GeoField
from multiprocessing import Pool
from component_analysis import pca_components_gf, orthomax, match_components_munkres,\
    matched_components
from geo_rendering import render_components, render_component_triple
from spatial_model_generator import constructVAR, make_model_geofield
from spca_meng import extract_sparse_components
from error_metrics import estimate_snr, mse_error, marpe_error
import mdp
from mdp.nodes import FastICANode

import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from surr_geo_field_ar import SurrGeoFieldAR



def estimate_components_ica(d):
    """
    Compute the ICA based on the input data d.
    """
    U, s, Vt = pca_components_gf(d, True)
    U = U[:, :NUM_COMPONENTS]
    V = np.transpose(Vt)
    V = V[:, :NUM_COMPONENTS]
    f = FastICANode(whitened = True, max_it = 10000, g = 'tanh', fine_g = 'tanh', max_it_fine = 1000)
    Vr = f.execute(V)
    P = f.get_projmatrix()
    Ur = np.dot(U, P)
    Ur /= np.sum(Ur**2, axis = 0) ** 0.5
    return Ur
    

def estimate_components_orthomax(d):
    """
    Compute the PCA/FA components based on the input data d
    as returned by GeoField bootstrap constructor.
    """
    U, s, _ = pca_components_gf(d)
    U = U[:, :NUM_COMPONENTS]
    if not ROTATE_NORMALIZED:
        U *= s[np.newaxis, :NUM_COMPONENTS]
    Ur, _, _ = orthomax(U, gamma = GAMMA, norm_rows=True)
    Ur /= np.sum(Ur**2, axis = 0) ** 0.5
    return Ur


def estimate_components_meng(d):
    """
    Compute components using the method of Meng.
    """
    U, _, _ = pca_components_gf(d)
    C = extract_sparse_components(U, SPCA_SPARSITY, NUM_COMPONENTS, U)
    return C



#
# Current simulation parameters
#
NUM_BOOTSTRAPS = 500
NUM_COMPONENTS = 3
POOL_SIZE = 3
RECOMPUTE_MODEL = True
GAMMA = 1.0
ROTATE_NORMALIZED = True
COMPONENT_ESTIMATOR = estimate_components_orthomax
SPCA_SPARSITY = 200


def compute_bootstrap_sample_components(x):
    gf, Urd = x
    
    # commo operation - generate new bootstrap sample
    b = gf.sample_temporal_bootstrap()
    
    # custom method to compute the components
    Ur = COMPONENT_ESTIMATOR(b)
    
    # match, flip sign and permute the discovered components    
    perm, sign_flip = match_components_munkres(Urd, Ur)
    Ur = Ur[:, perm]
    Ur *= sign_flip[:Ur.shape[1]]
    
    return Ur


def render_components_par(x):
    C, lats, lons, tmpl, ndx = x
    render_components(C, lats, lons, tmpl, ndx)

def render_triples_par(x):
    render_component_triple(*x)

def render_test_images(x):
    data, pltnames, file = x
    
    plt.figure(figsize = (10,len(data)*4))
    for i in range(len(data)):
        plt.subplot(len(data)*100 + 10 + i + 1)
        plt.imshow(data[i], interpolation = 'nearest')
        plt.title(pltnames[i])
        plt.colorbar()
        
    plt.savefig(file)



if __name__ == "__main__":

    print("Bootstrap analysis of uncertainty of VARIMAX components 1.0")
    
#    print("Loading data ...")
#    gf = GeoField()
#    gf.load("data/pres.mon.mean.nc", 'pres')
#    gf.transform_to_anomalies()
#    gf.normalize_monthly_variance()
#    gf.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
#    gf.slice_spatial(None, [20, 89])
#    gf.slice_months([12, 1, 2])

    # construct a test system    
    S = np.zeros(shape = (20, 50), dtype = np.int32)
    S[10:18, 25:45] = 1
    S[0:3, 6:12] = 2
    S[8:15, 2:12] = 3
    v, Sr = constructVAR(S, [0.0, 0.4, 0.8, 0.7], [-0.5, 0.5], [0.0, 0.0])
#    v, Sr = constructVAR(S, [0.0, 0.001, 0.01], [-0.1, 0.1], [0.00, 0.00], [0.01, 0.01])
    ts = v.simulate(200)
    gf = make_model_geofield(S, ts)
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)

    # replace field with surrogate field
    sgf = SurrGeoFieldAR()
    sgf.copy_field(gf)
    sgf.prepare_surrogates(pool)
    sgf.construct_surrogate_with_noise()
    gf = sgf
    gf.d = gf.surr_data().copy()
    
#    # construct "components" from the structural matrix
    Uopt = np.zeros((len(Sr), np.amax(Sr)))   
    for i in range(Uopt.shape[1]):
        Uopt[:,i] = np.where(Sr == (i+1), 1.0, 0.0)
        # remove the first element (it's the driver which is not included in the optimal component)
        Uopt[np.nonzero(Uopt[:,i])[0][0],i] = 0.0
        Uopt[:,i] /= np.sum(Uopt[:,i]**2) ** 0.5

    print("Analyzing data ...")
    
    # compute the eigenvalues and eigenvectors of the (spatial) covariance matrix 
    Ud, sd, Vtd = pca_components_gf(gf.data())
    Ud = Ud[:, :NUM_COMPONENTS]
    if not ROTATE_NORMALIZED:
        Ud *= sd[np.newaxis, :NUM_COMPONENTS]
        
    # estimate the components
    Ur = COMPONENT_ESTIMATOR(gf.data())
    
    print("Running bootstrap analysis [%d samples]" % NUM_BOOTSTRAPS)

    # initialize maximal and minimal boostraps
    max_comp = np.abs(Ur.copy())
    min_comp = np.abs(Ur.copy())
    mean_comp = np.zeros_like(Ur)
    var_comp = np.zeros_like(Ur)
    
    # generate and compute eigenvalues for 20000 surrogates
    t1 = datetime.now()
    
    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    print("Running parallel generation of bootstraps and SVD")
    slam_list = pool.map(compute_bootstrap_sample_components, [(gf, Ur)] * NUM_BOOTSTRAPS)
#    slam_list = map(compute_bootstrap_sample_components, [(gf, Ur)] * NUM_BOOTSTRAPS)

    num_comp = 0    
    for Urb in slam_list:
        max_comp = np.maximum(max_comp, np.abs(Urb))
        min_comp = np.minimum(min_comp, np.abs(Urb))

        # numerically robust computation of mean
        num_comp += 1
        delta = Urb - mean_comp
        mean_comp += delta / num_comp
        var_comp += delta * (Urb - mean_comp)
       
    var_comp /= (num_comp - 1)
    del slam_list
    
    # change variance to a small positive value to prevent NaN warning
    var_comp[var_comp == 0] = 1e-6
    
    # how do we now estimate the SNRs?
    # a. from data
    # b. from bootstrap means and stdevs -> T-values
    Urm = matched_components(Uopt, Ur)
#    Umvm = matched_components(Uopt, mean_comp / (var_comp ** 0.5))
    Umvm = matched_components(Uopt, mean_comp)
    
    print estimate_snr(Uopt, Urm)
    print estimate_snr(Uopt, Umvm)

    print mse_error(Uopt, Urm)
    print mse_error(Uopt, Umvm)
    
    print marpe_error(Uopt, Urm)
    print marpe_error(Uopt, Umvm)
    
    plt.figure(figsize = (12, 6))
    for i in range(3):

        plt.subplot(320+i*2+1)
        plt.plot(Uopt[:,i], 'r-')
        plt.plot(Umvm[:,i], 'b-')
        plt.plot(np.ones((Uopt.shape[0],1)) * (1.0 / Uopt.shape[0]**0.5), 'g-')
        plt.title('Component %d [bootstrap mean]' % i)
    
        plt.subplot(320+i*2+2)
        plt.plot(Uopt[:,i], 'r-')
        plt.plot(Urm[:,i], 'b-')
        plt.plot(np.ones((Uopt.shape[0],1)) * (1.0 / Uopt.shape[0]**0.5), 'g-')
        plt.title('Component %d [data]' % i)

    BUr = 1.0 / np.sum(Ur**2, axis = 0) ** 0.5
    Bmc = 1.0 / np.sum(mean_comp**2, axis = 0) ** 0.5
    
    Omc = Bmc[:, np.newaxis] * np.dot(mean_comp.T, mean_comp) * Bmc[np.newaxis, :]
    OUr = BUr[:, np.newaxis] * np.dot(Ur.T, Ur) * BUr[np.newaxis, :]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(OUr, interpolation = 'nearest')
    plt.title('Cross dots Ur')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(Omc, interpolation = 'nearest')
    plt.title('Cross dots mc')
    plt.colorbar()
    plt.show()
    
    # reshape all the fields back into correct spatial dimensions to match lon/lat of the original geofield
    Ud = gf.reshape_flat_field(Ud)
    Ur = gf.reshape_flat_field(Ur)
    max_comp = gf.reshape_flat_field(max_comp)
    min_comp = gf.reshape_flat_field(min_comp)
    mean_comp = gf.reshape_flat_field(mean_comp)
    var_comp = gf.reshape_flat_field(var_comp)

#    print("Rendering components in parallel ...")
#    render_list_job = [ ([Ur[i, ...], mean_comp[i,...], var_comp[i,...] ** 0.5, mean_comp[i,...] / (var_comp[i,...] ** 0.5)],
#                        ['Ur', 'Mean', 'Stdev', 'Mean/Stdev'],
#                        'figs/avg_test_component%02d.png' % (i+1)) for i in range(NUM_COMPONENTS)]
#    pool.map(render_test_images, render_list_job)
#
#    render_list_job = [ ([Ur[i, ...], Ud[i, ...], min_comp[i,...], max_comp[i,...] ** 0.5],
#                         ['Ur', 'PCA', 'Min', 'Max'],
#                         'figs/mm_test_component%02d.png' % (i+1)) for i in range(NUM_COMPONENTS)]
#    pool.map(render_test_images, render_list_job)

    print("DONE.")
