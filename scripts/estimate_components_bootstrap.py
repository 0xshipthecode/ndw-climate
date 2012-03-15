

from datetime import date, datetime
from geo_field import GeoField
from multiprocessing import Pool
from component_analysis import pca_eigvals, pca_components, varimax
from geo_rendering import render_components, render_component_triple
from spatial_model_generator import constructVAR, make_model_geofield

from munkres import Munkres
import os.path
import sys
import numpy as np
import pylab as pb
import cPickle

#
# Current simulation parameters
#
NUM_BOOTSTRAPS = 1000
NUM_COMPONENTS = 4
POOL_SIZE = None
RECOMPUTE_MODEL = True


def compute_bootstrap_sample_components(x):
    gf, Urd = x
    b = gf.sample_temporal_bootstrap()
    U, _, _ = pca_components(b)
    Ur, _, _ = varimax(U[:, :NUM_COMPONENTS])
    
    # compute closeness of components
    C = np.dot(Ur.T, Urd)
    
    # find optimal matching of components
    m = Munkres()
    match = m.compute(1.0 - np.abs(C))
    perm = np.zeros((NUM_COMPONENTS,), dtype = np.int)
    for i in range(len(match)):
        m_i = match[i]
        perm[m_i[0]] = m_i[1]
        Ur[:, m_i[1]] = -Ur[:, m_i[1]] if C[m_i[0], m_i[1]] < 0.0 else Ur[:, m_i[1]] 
        
    # reorder the bootstrap components according to the best matching 
    Ur = Ur[:, perm]
    
    return Ur


def render_components_par(x):
    C, lats, lons, tmpl, ndx = x
    render_components(C, lats, lons, tmpl, ndx)


def render_triples_par(x):
    render_component_triple(*x)

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
    v, Sr = constructVAR(S, [0.0, 0.001, 0.01], [-0.1, 0.1], [0.00, 0.00], [0.01, 0.01])
    ts = v.simulate(768)
    gf = make_model_geofield(S, ts)
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)
    
    print("Analyzing data ...")
    
    # compute the eigenvalues and eigenvectors of the (spatial) covariance matrix 
    Ud, sd, Vtd = pca_components(gf.data())
    Ud = Ud[:, :NUM_COMPONENTS]
    Ur, _, its = varimax(Ud)
    
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
    
    # reshape all the fields back into correct spatial dimensions to match lon/lat of the original geofield
    Ud = gf.reshape_flat_field(Ud)
    Ur = gf.reshape_flat_field(Ur)
    max_comp = gf.reshape_flat_field(max_comp)
    min_comp = gf.reshape_flat_field(min_comp)
    mean_comp = gf.reshape_flat_field(mean_comp)
    var_comp = gf.reshape_flat_field(var_comp)

    with open('results/bootstrap_results.bin', 'w') as f:
        cPickle.dump([Ud,Ur,max_comp,min_comp,mean_comp,var_comp], f)
    
#    render_list_triples = [ (Ur[i, ...], min_comp[i, ...], max_comp[i, ...], gf.lats, gf.lons, 'figs/nhemi_comp%02d_varimax.png' % (i+1), 'Component %d' % (i+1)) for i in range(NUM_COMPONENTS)]
    render_list_triples = [ (Ur[i, ...], mean_comp[i, ...], mean_comp[i, ...] / (var_comp[i, ...] ** 0.5) * (NUM_BOOTSTRAPS ** 0.5),
                             ['data', 'mean', 'T'], gf.lats, gf.lons, False,
                             'figs/nhemi_comp%02d_varimax.png' % (i+1),
                             'Component %d' % (i+1)) for i in range(NUM_COMPONENTS) ]

    pool.map(render_triples_par, render_list_triples)
    print("Rendering components in parallel ...")
   
    print("DONE.")
