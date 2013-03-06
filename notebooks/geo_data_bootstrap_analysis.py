# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from multiprocessing import Pool
from component_analysis import pca_components_gf, orthomax, match_components_munkres,\
    matched_components
from geo_rendering import render_components, render_component_triple, render_component_single
from spatial_model_generator import constructVAR, make_model_geofield
from spca_meng import extract_sparse_components
from error_metrics import estimate_snr, mse_error, marpe_error
from geo_data_loader import load_monthly_slp_all, load_monthly_sat_all, load_monthly_data_general
import mdp
from mdp.nodes import FastICANode

import os.path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random
from datetime import date, datetime
import time

# <codecell>

# Estimation methods

def estimate_components_ica(d):
    """
    Compute the ICA based on the input data d.
    """
    U, s, Vt = pca_components_gf(d)
    U = U[:, :NUM_COMPONENTS]
    V = np.transpose(Vt)
    V = V[:, :NUM_COMPONENTS]
    f = FastICANode(whitened = True, max_it = 10000, g = 'tanh', fine_g = 'tanh', max_it_fine = 1000)
    f.execute(V)
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
    Ur, _, iters = orthomax(U, rtol = np.finfo(np.float32).eps ** 0.5,
                            gamma = GAMMA,
                            maxiter = 500,
                            norm_rows = ROTATE_NORM_ROWS)
    Ur /= np.sum(Ur**2, axis = 0) ** 0.5
    if iters >= 499:
        print('Warning: max iters reached.')
        return None
    else:
        return Ur


def estimate_components_meng(d):
    """
    Compute components using the method of Meng.
    """
    U, _, _ = pca_components_gf(d)
    C = extract_sparse_components(U, SPCA_SPARSITY, NUM_COMPONENTS, U)
    return C

def estimate_components_spca(d):
    """
    Compute spatial PCA components.
    """
    U, _, _ = pca_components_gf(d)
    return U[:, :NUM_COMPONENTS]

def estimate_components_tpca(d):
    """
    Compute spatial PCA components.
    """
    U, _, _ = pca_components_gf(d, False)
    return U[:, :NUM_COMPONENTS]

# <markdowncell>

# The following set of parameters defines the computation.

# <codecell>

#
# Current simulation parameters
#
DISCARD_RATE = 0.2
BOOTSTRAP_STEP = 50
NUM_BOOTSTRAPS = 1000
NUM_COMPONENTS = 40
USE_SURROGATE_MODEL = False
COSINE_REWEIGHTING = True
MAX_AR_ORDER = 30
POOL_SIZE = 20
RECOMPUTE_MODEL = True
GAMMA = 1.0
ROTATE_NORMALIZED = True
ROTATE_NORM_ROWS = False
COMPONENT_ESTIMATOR = estimate_components_orthomax
SPCA_SPARSITY = 200

random.seed()
np.random.seed()

# <codecell>

def compute_bootstrap_sample_components(x):
    gf, Urd = x
    
    Nc = Urd.shape[1]
    
    # common operation - generate new bootstrap sample
    b = gf.sample_temporal_bootstrap()
    
    if COSINE_REWEIGHTING:
        b *= gf.qea_latitude_weights()
    
    # custom method to compute the components
    Ur = COMPONENT_ESTIMATOR(b)
    if Ur is None:
        return None
    
    # match, flip sign and permute the discovered components    
    perm, sign_flip = match_components_munkres(Urd, Ur)
    Ur = Ur[:, perm[:Nc]]
    Ur *= sign_flip[:, :Nc]
    
    return Ur

# <codecell>

# Rendering functions
#
def render_components_par(x):
    C, lats, lons, tmpl, ndx = x
    render_components(C, lats, lons, tmpl, ndx)

def render_triples_par(x):
    render_component_triple(*x)
    plt.close()
    print("Rendered triple [%s]" % x[-1])


def render_test_images(x):
    data, pltnames, file = x
    
    plt.figure(figsize = (10,len(data)*4))
    for i in range(len(data)):
        plt.subplot(len(data)*100 + 10 + i + 1)
        plt.imshow(data[i], interpolation = 'nearest')
        plt.title(pltnames[i])
        plt.colorbar()
        
    plt.savefig(file)

# <codecell>

# change to higher dir for loading to work
os.chdir('/home/martin/Projects/Climate/ndw-climate/')

# load up the monthly SLP geo-field
# print("[%s] Loading SLP geo field..." % (str(datetime.now())))
# gf = load_monthly_slp_all()

print("[%s] Loading SAT SH geo field..." % (str(datetime.now())))
gf = load_monthly_data_general('data/air.mon.mean.nc', 'air', date(1948, 1, 1), date(2012, 1, 1), None, None, [-89, 0], 0)

# load up the monthly SAT geo-field
# print("[%s] Loading SAT geo field..." % (str(datetime.now())))
# gf = load_monthly_sat_all()
print("[%s] Field loaded." % (str(datetime.now())))

# <codecell>

print gf.d.shape
print gf.lons[0], gf.lons[-1]
print gf.lats[0], gf.lats[-1]
print gf.d.shape[1] * gf.d.shape[2]

# <codecell>

if USE_SURROGATE_MODEL:
    pool = Pool(POOL_SIZE)
    sgf = SurrGeoFieldAR([0, MAX_AR_ORDER], 'sbc')
    print("Running preparation of surrogates ...")
    sgf.copy_field(gf)
    sgf.prepare_surrogates(pool)
    sgf.construct_surrogate_with_noise()
    sgf.d = sgf.sd # hack to replace original data with surrogate
    print("Max AR order is %d ..." % sgf.max_ord)
    gf = sgf
    print("Replaced field with surrogate field.")
    pool.close()
    del pool
    
print("Analyzing data ...")
d = gf.data()
if COSINE_REWEIGHTING:
    d *= gf.qea_latitude_weights()
Ud, sd, Vtd = pca_components_gf(d)
Ud = Ud[:, :NUM_COMPONENTS]
if not ROTATE_NORMALIZED:
    Ud *= sd[np.newaxis, :NUM_COMPONENTS]
    
# estimate the components
Ur = COMPONENT_ESTIMATOR(d)
print("DONE.")

# <codecell>

print(np.sum(sd[:NUM_COMPONENTS]) / np.sum(sd))
print(1.0*NUM_COMPONENTS/len(sd))

# <codecell>

# rychly test vysledku
#for i in range(5):
#    tmp = gf.reshape_flat_field(Ur[:, i:i+1])
#    render_component_single(tmp[0, :, :], gf.lats, gf.lons, False, None, 'QEA cosw component %d' % (i+1))
#del tmp

# <codecell>

t_start = datetime.now()

# initialize a parallel pool
pool = Pool(POOL_SIZE)

print("Running bootstrap analysis [%d samples] at %s" % (NUM_BOOTSTRAPS, str(t_start)))
# initialize maximal and minimal boostraps
EXTREMA_MEMORY = ceil(DISCARD_RATE * NUM_BOOTSTRAPS)
max_comp = np.tile(np.abs(Ur.copy()), (EXTREMA_MEMORY + BOOTSTRAP_STEP, 1, 1))
min_comp = np.tile(np.abs(Ur.copy()), (EXTREMA_MEMORY + BOOTSTRAP_STEP, 1, 1))
mean_comp = np.zeros_like(Ur)
var_comp = np.zeros_like(Ur)
    
bsmp_done = 0
divergent_computations = 0
print("Running parallel generation of %d bootstraps." % NUM_BOOTSTRAPS)
while bsmp_done < NUM_BOOTSTRAPS:
    
    t1 = datetime.now()
    bsmp_todo_now = min(NUM_BOOTSTRAPS - bsmp_done, BOOTSTRAP_STEP)

    print("STAGE START [%s]: working on %d bootstraps." % (str(t1), bsmp_todo_now))

    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    slam_list = pool.map(compute_bootstrap_sample_components, [(gf, Ur)] * bsmp_todo_now)
    
    t1a = datetime.now()

    # third sorting strategy (for large bootstrap samples) - insert in everything that was in this step
    # and sort only once
    i = 0
    for Urb in slam_list:
        
        # check if the result is valid
        if Urb is None:
            divergent_computations += 1
            continue
            
        min_comp[EXTREMA_MEMORY+i, :, :] = np.abs(Urb)
        max_comp[BOOTSTRAP_STEP-i-1, :, :] = np.abs(Urb)
        
        bsmp_done += 1
        delta = Urb - mean_comp
        mean_comp += delta / bsmp_done
        var_comp += delta * (Urb - mean_comp)
        i += 1
        
    # sort the entries along first axis
    min_comp[:EXTREMA_MEMORY + bsmp_todo_now, :, :].sort(axis = 0)
    max_comp[BOOTSTRAP_STEP - bsmp_todo_now:, :, :].sort(axis = 0)
    
    # print progress based on rate of completion from beginning of computation (from t_start)
    t2 = datetime.now()
    dt = (t2 - t_start) * (NUM_BOOTSTRAPS - bsmp_done) // bsmp_done
    
    da = t2 - t1a
    db = t2 - t1
        
    # print progress
    print("PROGRESS [%s]: %d/%d complete (sorting %G percent), predicted completion at %s [total divergents %d]." % 
          (str(t2), bsmp_done, NUM_BOOTSTRAPS,
           int((da.seconds + da.microseconds/10e6) / (db.seconds + db.microseconds/10e6) * 100),
           str(t2 + dt), divergent_computations))

var_comp /= (bsmp_done - 1)

print("DONE at %s after %s" % (str(datetime.now()), str(datetime.now() - t_start)))

pool.close()

max_comp = max_comp[BOOTSTRAP_STEP, :, :]
min_comp = min_comp[EXTREMA_MEMORY, :, :]

# save the results to file
with open('results/sat_sh_var_bootstrap_results_b1000_cosweights_varimax.bin', 'w') as f:
    cPickle.dump({ 'Ud' : Ud, 'Ur' : Ur, 'max' : max_comp, 'min' : min_comp,
                  'mean' : mean_comp, 'var' : var_comp, 'dlambda' : sd}, f)

# <codecell>

#load results from file
#with open('../results/slp_nh_var_surrogate_bootstrap_results_b1000_cosweights.bin', 'r') as f:
#    d = cPickle.load(f)
#Ud = d['Ud']
#Ur = d['Ur']
#max_comp = d['max']
#min_comp = d['min']
#mean_comp = d['mean']
#var_comp = d['var']
#del d

# <codecell>

# reshape all the fields back into correct spatial dimensions to match lon/lat of the original geofield
max_comp_gf = gf.reshape_flat_field(max_comp)
min_comp_gf = gf.reshape_flat_field(min_comp)
mean_comp_gf = gf.reshape_flat_field(mean_comp)
var_comp_gf = gf.reshape_flat_field(var_comp)

# Ud and Ur are reshaped into new variables as the old ones may be reused in a new computation
Ud_gf = gf.reshape_flat_field(Ud)
Ur_gf = gf.reshape_flat_field(Ur)

# <codecell>

BUr = 1.0 / np.sum(Ur**2, axis = 0) ** 0.5
Bmc = 1.0 / np.sum(mean_comp**2, axis = 0) ** 0.5

Omc = Bmc[:, np.newaxis] * np.dot(mean_comp.T, mean_comp) * Bmc[np.newaxis, :]
OUr = BUr[:, np.newaxis] * np.dot(Ur.T, Ur) * BUr[np.newaxis, :]

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.imshow(OUr - np.eye(NUM_COMPONENTS), interpolation = 'nearest')
plt.title('Cross dots Ur')
plt.colorbar()
plt.subplot(122)
plt.imshow(Omc - np.eye(NUM_COMPONENTS), interpolation = 'nearest')
plt.title('Cross dots mc')
plt.colorbar()
plt.show()

# <codecell>

t_start = datetime.now()
print("Rendering components in parallel at [%s] ..." % str(t_start))
render_list_triples = [ (Ur_gf[i, ...],
                         min_comp_gf[i, ...],
                         max_comp_gf[i, ...],
                         [ 'Data', 'Min', 'Max' ], 
                         gf.lats, gf.lons, True,
                         '../figs/nhemi_comp%02d_varimax_rough.png' % (i+1),
                         'Component %d' % (i+1))
                                 for i in range(NUM_COMPONENTS) ]

# use less nodes to render the maps due to memory constraints
pool = Pool(POOL_SIZE)
pool.map(render_triples_par, render_list_triples)
pool.close()

# clear some memory
del render_list_triples

print("DONE after [%s]." % (datetime.now() - t_start))

# <codecell>

t_start = datetime.now()
thr = 1.0 / sqrt(mean_comp.shape[0])

render_list_triples = [ (mean_comp_gf[i, ...], Ur_gf[i, ...], 
#                        mean_comp_gf[i, ...] / (var_comp_gf[i,...] ** 0.5 + 0.01) * (NUM_BOOTSTRAPS ** 0.5), 
                         mean_comp_gf[i, ...] * (np.abs(mean_comp_gf[i,...]) > thr),
                        [ 'Mean', 'Data', 'Mean:Thr'],
                        gf.lats, gf.lons, False,
                        '../figs/nhemi_comp%02d_varimax_mn_cos_nnorm.png' % (i+1),
                        'Component %d' % (i+1))
                                for i in range(NUM_COMPONENTS)]

# use less nodes to render the maps due to memory constraints
pool = Pool(POOL_SIZE)
pool.map(render_triples_par, render_list_triples)
pool.close()

# clean up memory
del render_list_triples

print("DONE after [%s]." % (datetime.now() - t_start))

