# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from component_analysis import pca_components_gf, orthomax, match_components_munkres, matched_components
from geo_rendering import render_components, render_component_triple, render_component_single
from spatial_model_generator import constructVAR, make_model_geofield
from spca_meng import extract_sparse_components
from error_metrics import estimate_snr, mse_error, marpe_error
from geo_data_loader import load_monthly_slp_all, load_monthly_sat_all, load_monthly_hgt500_all
from multiprocessing import Process, Queue, Pool

import math
import os.path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random
from datetime import date, datetime
import time

# open the log file and create a closure that writes to it
log_file = open('geodata_bootstrap_components-%s.log' % datetime.now().strftime('%Y%m%d-%H%M'), 'w')
def log(msg):
    log_file.write('[%s] %s\n' % (str(datetime.now()), msg))
    log_file.flush()

# Component computation methods

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
    try:
        U, s, _ = pca_components_gf(d)
        U = U[:, :NUM_COMPONENTS]
        if not ROTATE_NORMALIZED:
            U *= s[np.newaxis, :NUM_COMPONENTS]
        Ur, T, iters = orthomax(U,
                                rtol = np.finfo(np.float32).eps ** 0.5,
                                gamma = GAMMA,
                                maxiter = 500,
                                norm_rows = ROTATE_NORM_ROWS)
        Ur /= np.sum(Ur**2, axis = 0) ** 0.5
        if iters >= 499:
            return None
        else:
            return Ur, T
    except LinAlgError as e: 
        print("**LINALG ERROR** code: %d text : %s" % (e.errno, e.strerror))
    except:
        print("**UNEXPECTED ERROR** %s" % sys.exc_info()[0])


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
NUM_BOOTSTRAPS = 100
NUM_COMPONENTS = 59
USE_SURROGATE_MODEL = False
COSINE_REWEIGHTING = True
MAX_AR_ORDER = 30
WORKER_COUNT = 16
GAMMA = 1.0
ROTATE_NORMALIZED = True
ROTATE_NORM_ROWS = False
COMPONENT_ESTIMATOR = estimate_components_orthomax
SPCA_SPARSITY = 200
DETREND = False
DATA_NAME = 'slp_all'
SUFFIX =""

# write all settings to log file
log('Analyzing data: %s with suffix: %s' % (DATA_NAME, SUFFIX))
log('NBootstraps: %d NComps: %d UseSurrModel: %s CosWeight: %s Detrend: %s' % 
          (NUM_BOOTSTRAPS, NUM_COMPONENTS, USE_SURROGATE_MODEL, COSINE_REWEIGHTING, DETREND))
log('Gamma: %g RotNorm: %s RotNormRows: %s' %
          (GAMMA, ROTATE_NORMALIZED, ROTATE_NORM_ROWS))


loader_functions = {
    'sat_all' : load_monthly_sat_all,
    'slp_all' : load_monthly_slp_all,
    'hgt500_all' : load_monthly_hgt500_all
}

np.random.seed()

# <codecell>


def compute_bootstrap_sample_components(gf, Urd, jobq, resq):
    """
    Estimate the components from a temporally bootstrapped dataset.
    """
    Nc = Urd.shape[1]
    while jobq.get() is not None:
    
        # common operation - generate new bootstrap sample
        b, ndx = gf.sample_temporal_bootstrap()
    
        if COSINE_REWEIGHTING:
            b *= gf.qea_latitude_weights()
    
        # custom method to compute the components
        Res = COMPONENT_ESTIMATOR(b)
        if Res is None:
            resq.put(None)
        else:
            Ur, _ = Res
            # match, flip sign and permute the discovered components    
            perm, sign_flip = match_components_munkres(Urd, Ur)
            Ur = Ur[:, perm[:Nc]]
            Ur *= sign_flip[:, :Nc]
            resq.put(Ur)


# change to higher dir for loading to work
os.chdir('/home/martin/Projects/Climate/ndw-climate/')

log("Loading geo field %s ..." % DATA_NAME)
gf = loader_functions[DATA_NAME]()

if USE_SURROGATE_MODEL:
    pool = Pool(WORKER_COUNT)
    sgf = SurrGeoFieldAR([0, MAX_AR_ORDER], 'sbc')
    log("Running preparation of surrogates ...")
    sgf.copy_field(gf)
    sgf.prepare_surrogates(pool)
    sgf.construct_surrogate_with_noise()
    sgf.d = sgf.sd # hack to replace original data with surrogate
    log("Max AR order is %d ..." % sgf.max_ord)
    gf = sgf
    log("Replaced field with surrogate field.")
    pool.close()
    del pool

log("Analyzing data ...")
d = gf.data()
if COSINE_REWEIGHTING:
    d *= gf.qea_latitude_weights()
Ud, sd, Vtd = pca_components_gf(d)
Ud = Ud[:, :NUM_COMPONENTS]
if not ROTATE_NORMALIZED:
    Ud *= sd[np.newaxis, :NUM_COMPONENTS]
    
log("Total variance %g explained by selected components %g." % (np.sum(sd[:NUM_COMPONENTS]),
                                                                np.sum(sd[:NUM_COMPONENTS]) / np.sum(sd)))

# estimate the components and their variance
Ur, T = COMPONENT_ESTIMATOR(d)
T = np.matrix(T)
expvar = np.diag(np.transpose(T) * np.matrix(np.diag(sd[:NUM_COMPONENTS])) * T)

# prepare parallel run
jobq = Queue()
resq = Queue()
for i in range(NUM_BOOTSTRAPS):
    jobq.put(1)
for i in range(WORKER_COUNT):
    jobq.put(None)

log("Starting workers")
workers = [Process(target = compute_bootstrap_sample_components, args = (gf, Ur, jobq, resq)) for i in range(WORKER_COUNT)]

EXTREMA_MEMORY = math.ceil(DISCARD_RATE * NUM_BOOTSTRAPS)
max_comp = np.tile(np.abs(Ur.copy()), (EXTREMA_MEMORY + 1, 1, 1))
min_comp = np.tile(np.abs(Ur.copy()), (EXTREMA_MEMORY + 1, 1, 1))
mean_comp = np.zeros_like(Ur)
var_comp = np.zeros_like(Ur)
    
# start all workers (who immediately start processing the job queue)
for w in workers:
    w.start()

t_start = datetime.now()
t_last = t_start
log("Running parallel bootstrap with %d iterations at %s" % (NUM_BOOTSTRAPS, str(t_start)))

bsmp_done = 0
divergent_computations = 0
while bsmp_done < NUM_BOOTSTRAPS:
    
    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    Urb = resq.get()
    if Urb is None:
        divergent_computations += 1
        continue
    else:
        min_comp[EXTREMA_MEMORY, :, :] = np.abs(Urb)
        max_comp[0, :, :] = np.abs(Urb)
        
        bsmp_done += 1
        delta = Urb - mean_comp
        mean_comp += delta / bsmp_done
        var_comp += delta * (Urb - mean_comp)
        

    # sort the entries along first axis including the newly added item
    min_comp.sort(axis = 0)
    max_comp.sort(axis = 0)
    
    # predict time to go
    t_now = datetime.now()
        
    # print progress
    if (t_now - t_last).total_seconds() > 300:
        t_last = t_now
        dt = (t_now - t_start) / bsmp_done * (NUM_BOOTSTRAPS - bsmp_done)
        log("PROGRESS: %d/%d complete, predicted completion at %s, %d total div. computations."
            % (bsmp_done, NUM_BOOTSTRAPS, str(t_now + dt), divergent_computations))


# wait for all workers to finish
for w in workers:
    w.join()

var_comp /= (bsmp_done - 1)

log("DONE after %s with %d divergents, now saving data" % (str(datetime.now() - t_start), divergent_computations))

max_comp = max_comp[1, :, :]
min_comp = min_comp[EXTREMA_MEMORY, :, :]

# reorder all elements according to explained variance (descending)
nord = np.argsort(expvar)[::-1]
Ud = Ud[:, nord]
Ur = Ur[:, nord]
expvar = expvar[nord]
sd = sd[nord]
max_comp = max_comp[:,nord]
min_comp = min_comp[:,nord]
mean_comp = mean_comp[:,nord]
var_comp = var_comp[:,nord]

# save the results to file
filename = 'results/%s_var_b1000_cosweights_varimax%s.bin' % (DATA_NAME,SUFFIX)
with open(filename, 'w') as f:
    cPickle.dump({ 'Ud' : Ud, 'Ur' : Ur, 'max' : max_comp, 'min' : min_comp,
                   'mean' : mean_comp, 'var' : var_comp, 'dlam' : sd, 'expvar' : expvar,
                   'lats' : gf.lats, 'lons' : gf.lons}, f)

# Data was saved to file.
log("Data saved to file %s." % filename)
log_file.close()
