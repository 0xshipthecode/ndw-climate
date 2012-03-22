

from datetime import date, datetime
from geo_field import GeoField
from multiprocessing import Pool
from component_analysis import pca_eigvals, pca_components, orthomax
from geo_rendering import render_components, render_component_triple
from spatial_model_generator import constructVAR, make_model_geofield

from munkres import Munkres
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle

#
# Current simulation parameters
#
NUM_BOOTSTRAPS = 100
NUM_COMPONENTS = 3
POOL_SIZE = None
RECOMPUTE_MODEL = True
GAMMA = 1.0
ROTATE_NORMALIZED = False


def match_components_munkres(U1, U2):
    """
    Match the components from U1 to the components in U2 using the
    Hungarian algorithm.  The function returns a sign_flip vector
    which can be applied to U2 to switch the signs of the components to match
    those in U1.  The sign_flip, if applied, must be applied to U2 BEFORE
    the permutation!  The permutation which will bring U2 to match U1 is returned
    as the first element in the tuple.  Then U1 === U2[:, perm].
    
    synopsis: perm, sf = match_components_munkres(U1, U2)
        
    """
    NC = U2.shape[1]
    
    # compute closeness of components using the dot product
    C = np.dot(U1.T, U2)

    # normalize dot product matrix by sizes (to compare unit size vectors)    
    U1s = 1.0 / np.sum(U1**2, axis = 0) ** 0.5
    U2s = 1.0 / np.sum(U2**2, axis = 0) ** 0.5
    C = U1s[:, np.newaxis] * C * U2s[np.newaxis, :]
    
    # find optimal matching of components
    m = Munkres()
    match = m.compute(1.0 - np.abs(C))
    perm = np.zeros((NC,), dtype = np.int)
    sign_flip = np.zeros((1, NC), dtype = np.int)
    for i in range(len(match)):
        m_i = match[i]
        perm[m_i[0]] = m_i[1]
        sign_flip[0, m_i[1]] = -1 if C[m_i[0], m_i[1]] < 0.0 else 1.0
    
    return perm, sign_flip 


def estimate_snr(Sr, U):
    """
    Estimate compute the difference between means of signal and noise divided by the
    sum of standard deviations of the signal and noise for each component.  Sr is the
    structural matrix ravel()ed.
    """
    N, M = U.shape
    C = np.amax(Sr)
    S = np.zeros((N, C))
    
    # copy U, we will have to manipulate it
    U = U.copy()

    # construct "components" from the structural matrix   
    for i in range(C):
        S[:,i] = np.where(Sr == (i+1), 1.0, 0.0)
        # remove the first element (it's the driver which is not included in the testing)
        S[np.nonzero(S[:,i])[0][0],i] = 0.0
        S[:,i] /= np.sum(S[:,i]**2) ** 0.5

    # find the matching (even if there are more components in U)
    perm, sf = match_components_munkres(S, U)
    U *= sf
    U = U[:, perm[:C]]
    
    # compute the snr for each component (which is now matched according to index i)
    snr = np.zeros((C,))
    for i in range(C):
        mean_sig = np.mean(U[Sr == (i+1), i])
        std_sig = np.std(U[Sr == (i+1), i])
        mean_noise = np.mean(U[Sr != (i+1), i])
        std_noise = np.std(U[Sr != (i+1), i])
        snr[i] = (mean_sig - mean_noise) / (std_sig + std_noise)
        
    return snr


def compute_bootstrap_sample_components(x):
    gf, Urd = x
    b = gf.sample_temporal_bootstrap()
    U, s, _ = pca_components(b)
    U = U[:, :NUM_COMPONENTS]
    if not ROTATE_NORMALIZED:
        U *= s[np.newaxis, :NUM_COMPONENTS]
    Ur, _, _ = orthomax(U, gamma = GAMMA)

    # match, flip sign and permute the discovered components    
    perm, sign_flip = match_components_munkres(Urd, Ur)
    Ur *= sign_flip
    Ur = Ur[:, perm]
    
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
    
    print("Loading data ...")
#    gf = GeoField()
#    gf.load("data/pres.mon.mean.nc", 'pres')
#    gf.transform_to_anomalies()
#    gf.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
#    gf.slice_spatial(None, [20, 89])
#    gf.slice_months([12, 1, 2])
    
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
    
    print("Analyzing data ...")
    
    # compute the eigenvalues and eigenvectors of the (spatial) covariance matrix 
    Ud, sd, Vtd = pca_components(gf.data())
    Ud = Ud[:, :NUM_COMPONENTS]
    if not ROTATE_NORMALIZED:
        Ud *= sd[np.newaxis, :NUM_COMPONENTS]
    Ur, _, its = orthomax(Ud, gamma = GAMMA)
    
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

    num_comp = 0    
    for Urb in slam_list:
        max_comp = np.maximum(max_comp, np.abs(Urb))
        min_comp = np.minimum(min_comp, np.abs(Urb))

        num_comp += 1
        delta = Urb - mean_comp
        mean_comp += delta / num_comp
        var_comp += delta * (Urb - mean_comp)
       
    var_comp /= (num_comp - 1)
    del slam_list
    
    
    # how do we now estimate the SNRs?
    # a. from data
    # b. from bootstrap means and stdevs -> T-values
    snr_data = estimate_snr(S.ravel(), Ur)
    snr_bs_t = estimate_snr(S.ravel(), mean_comp / (var_comp ** 0.5))
    snr_bs_m = estimate_snr(S.ravel(), mean_comp)
    
    print snr_data
    print snr_bs_m
    print snr_bs_t
    
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
    render_list_job = [ ([Ur[i, ...], mean_comp[i,...], var_comp[i,...] ** 0.5, mean_comp[i,...] / (var_comp[i,...] ** 0.5)],
                        ['Ur', 'Mean', 'Stdev', 'Mean/Stdev'],
                        'figs/avg_test_component%02d.png' % (i+1)) for i in range(NUM_COMPONENTS)]
    pool.map(render_test_images, render_list_job)

    render_list_job = [ ([Ur[i, ...], Ud[i, ...], min_comp[i,...], max_comp[i,...] ** 0.5],
                         ['Ur', 'PCA', 'Min', 'Max'],
                         'figs/mm_test_component%02d.png' % (i+1)) for i in range(NUM_COMPONENTS)]
    pool.map(render_test_images, render_list_job)

    print("DONE.")
