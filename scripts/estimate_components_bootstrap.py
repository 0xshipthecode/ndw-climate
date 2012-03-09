

from datetime import date, datetime
from geo_field import GeoField
from multiprocessing import Pool
from component_analysis import pca_eigvals, pca_components, varimax
from geo_rendering import render_components, render_component_triple
from munkres import Munkres
import os.path
import sys
import numpy as np
import pylab as pb
import cPickle

#
# Current simulation parameters
#
NUM_BOOTSTRAPS = 16
NUM_COMPONENTS = 27
POOL_SIZE = None
RECOMPUTE_MODEL = True


def compute_bootstrap_sample_components(x):
    gf, Urd = x
    b = gf.sample_temporal_bootstrap()
    U, _, _ = pca_components(b)
    Ur, _, _ = varimax(U[:, :NUM_COMPONENTS])
    
    # compute closeness of components
    C = np.corrcoef(Ur, Urd, rowvar = 0)
    C = C[NUM_COMPONENTS:, :NUM_COMPONENTS]
    
    # find optimal matching of components
    m = Munkres()
    match = m.compute(1.0 - np.abs(C))
    perm = np.zeros((NUM_COMPONENTS,), dtype = np.int)
    signflip = np.ones((1, NUM_COMPONENTS))
    for i in range(len(match)):
        m_i = match[i]
        perm[m_i[0]] = m_i[1]
        signflip[0,i] = -1.0 if C[m_i[0], m_i[1]] < 0.0 else 1.0
        
    # reorder the bootstrap components according to the best matching 
    Ur = Ur[:, perm]
    Ur = Ur * signflip
    
    return Ur


def render_components_par(x):
    C, lats, lons, tmpl, ndx = x
    render_components(C, lats, lons, tmpl, ndx)


def render_triples_par(x):
    Cd, Cmn, Cmx, lats, lons, fname, pltname = x
    render_component_triple(Cd, Cmn, Cmx, lats, lons, fname, pltname)

if __name__ == "__main__":

    print("Bootstrap analysis of uncertainty of VARIMAX components 1.0")
    
    print("Loading data ...")
    gf = GeoField()
    gf.load("data/pres.mon.mean.nc", 'pres')
    gf.transform_to_anomalies()
    gf.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
    gf.slice_spatial(None, [20, 89])
    
    # initialize a parallel pool
    pool = Pool(POOL_SIZE)
    
    print("Analyzing data ...")
    
    # compute the eigenvalues and eigenvectors of the (spatial) covariance matrix 
    Ud, sd, Vtd = pca_components(gf.data())[:NUM_COMPONENTS]
    Ud = Ud[:, :NUM_COMPONENTS]
    Ur, _, its= varimax(Ud)
    
    print("Running bootstrap analysis [%d samples]" % NUM_BOOTSTRAPS)

    # initialize maximal and minimal boostraps
    max_comp = np.abs(Ur.copy())
    min_comp = np.abs(Ur.copy())
    
    # generate and compute eigenvalues for 20000 surrogates
    t1 = datetime.now()
    
    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    print("Running parallel generation of bootstraps and SVD")
    slam_list = pool.map(compute_bootstrap_sample_components, [(gf, Ur)] * NUM_BOOTSTRAPS)
    
    for Urb in slam_list:
        Urb = np.abs(Urb)
        max_comp = np.maximum(max_comp, Urb)
        min_comp = np.minimum(min_comp, Urb)
    del slam_list
    
    # reshape all the fields back into correct spatial dimensions to match lon/lat of the original geofield
    Ud = gf.reshape_flat_field(Ud)
    Ur = gf.reshape_flat_field(Ur)
    max_comp = gf.reshape_flat_field(max_comp)
    min_comp = gf.reshape_flat_field(min_comp)
    
    print("Rendering components in parallel ...")
    render_list_triples = [ (Ur[i, ...], min_comp[i, ...], max_comp[i, ...], gf.lats, gf.lons, 'figs/nhemi_comp%02d_varimax.png' % (i+1), 'Component %d' % (i+1)) for i in range(NUM_COMPONENTS)]
    pool.map(render_triples_par, render_list_triples)
   
    print("DONE.")
