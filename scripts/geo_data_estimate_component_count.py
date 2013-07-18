# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

from datetime import date, datetime
from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from multiprocessing import Process, Queue, Pool
from component_analysis import pca_eigvals_gf, pca_components_gf, matched_components, orthomax
from spatial_model_generator import constructVAR, make_model_geofield
from geo_data_loader import load_monthly_sat_all, load_monthly_slp_all, load_monthly_hgt500_all
from geo_rendering import render_component_single
from multi_stats import compute_eigvals_pvalues, fdr_test, bonferroni_test, holm_test

import os.path
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import scipy.signal as sps

#
# Current simulation parameters
#
NUM_SURR = 20000
USE_MUVAR = False
USE_SURROGATE_MODEL = False
COSINE_REWEIGHTING = True
NUM_EIGVALS = 100
WORKER_COUNT = 4
MAX_AR_ORDER = 30
DETREND = True
DATA_NAME = 'slp_all'
SUFFIX ="_detrended"


loader_functions = {
    'sat_all' : load_monthly_sat_all,
    'slp_all' : load_monthly_slp_all,
    'hgt500_all' : load_monthly_hgt500_all
}

np.random.seed()

# open the log file and create a closure that writes to it
log_file = open('geodata_estimate_component_count-%s.log' % datetime.now().strftime('%Y%m%d-%H%M'), 'w')
def log(msg):
    log_file.write('[%s] %s\n' % (str(datetime.now()), msg))
    log_file.flush()

log('Analyzing data: %s with suffix: %s' % (DATA_NAME, SUFFIX))
log('Nsurr: %d Muvar: %s UseSurrModel: %s CosWeight: %s NumEigvals: %d Detrend: %s' % 
          (NUM_SURR, USE_MUVAR, USE_SURROGATE_MODEL, COSINE_REWEIGHTING, NUM_EIGVALS, DETREND))

os.chdir('/home/martin/Projects/Climate/ndw-climate/')
log("Loading geo field...")

# this function loads the data and does SOME preprocessing on it,
# examine the function in src/geo_data_loader.py to see exactly what it's doing
gf = loader_functions[DATA_NAME]()

# if detrend is required, do it now
if DETREND:
    gf.detrend()


# If substitution of original data with multivariate surrogate is required, do it now
if USE_MUVAR:
    log("Constructing F-2 surrogate ...")
    sgf = SurrGeoFieldAR()
    sgf.copy_field(gf)
    sgf.construct_fourier2_surrogates()
    sgf.d = sgf.sd.copy()

    # slide in fourier2 surrogate
    log("replacing data with surrogate ...")
    orig_gf = gf
    gf = sgf

log("Done loading, data has shape %s." % str(gf.d.shape))


def compute_surrogate_cov_eigvals(sd, jobq, resq):
    while jobq.get() is not None:

        # construct AR/SBC surrogates
        sd.construct_surrogate_with_noise()
        d = sd.surr_data()
        if COSINE_REWEIGHTING:
            d *= sd.qea_latitude_weights()
        sm_ar = pca_eigvals_gf(d)
        sm_ar = sm_ar[:NUM_EIGVALS]
    
        # construct fourier surrogates
        sd.construct_fourier1_surrogates()
        d = sd.surr_data()
        if COSINE_REWEIGHTING:
            d *= sd.qea_latitude_weights()
        sm_f = pca_eigvals_gf(d)
        sm_f = sm_f[:NUM_EIGVALS]

        # shuffle data (white noise surrogates)
        d = sd.data()
        N = d.shape[0]
        for i in range(d.shape[1]):
            for j in range(d.shape[2]):
                ndx = np.argsort(np.random.normal(size = (N,)))
                d[:, i, j] = d[ndx, i, j]
        if COSINE_REWEIGHTING:
            d = d * sd.qea_latitude_weights()
        sm_w1 = pca_eigvals_gf(d)
        sm_w1 = sm_w1[:NUM_EIGVALS]
    
        resq.put((sm_ar, sm_w1, sm_f))


log("Constructing surrogate model ...")
sgf = SurrGeoFieldAR([0, MAX_AR_ORDER], 'sbc')
sgf.copy_field(gf)
pool = Pool(WORKER_COUNT)
sgf.prepare_surrogates(pool)
sgf.construct_surrogate_with_noise()
if pool is not None:
    pool.close()
    pool.join()
    del pool

log("Surrogate model constructed.")

if USE_SURROGATE_MODEL:
    # HACK to replace original data with surrogates
    gf.d = sgf.sd.copy()
    sgf.d = sgf.sd.copy()
    log("** WARNING ** Replaced synth model with surrogate model to check false positives.")

# analyze data & obtain eigvals and surrogates
log("Computing eigenvalues of dataset ...")
d = gf.data()
if COSINE_REWEIGHTING:
    d *= gf.qea_latitude_weights()
dlam = pca_eigvals_gf(d)[:NUM_EIGVALS]
    
log("Rendering orders of fitted AR models.")
mo = sgf.model_orders()
render_component_single(mo, gf.lats, gf.lons, plt_name = 'Model orders of AR surrogates',
                        fname='%s_ar_model_order%s.png' % (DATA_NAME, SUFFIX))

# storage for three types of surrogates
slam_ar = np.zeros((NUM_SURR, NUM_EIGVALS))
slam_w1 = np.zeros((NUM_SURR, NUM_EIGVALS))
slam_f = np.zeros((NUM_SURR, NUM_EIGVALS))

surr_completed = 0

# construct the job queue

jobq = Queue()
resq = Queue()
for i in range(NUM_SURR):
    jobq.put(1)
for i in range(WORKER_COUNT):
    jobq.put(None)

log("Starting workers")
workers = [Process(target = compute_surrogate_cov_eigvals,
                   args = (sgf,jobq,resq)) for i in range(WORKER_COUNT)]

# construct the surrogates in parallel
# we can duplicate the list here without worry as it will be copied into new python processes
# thus creating separate copies of sd
log("Running parallel generation of surrogates and analysis.")

# generate and compute eigenvalues for 20000 surrogates
t_start = datetime.now()

# start all workers (who immediately start processing the job queue)
for w in workers:
    w.start()

surr_completed = 0
t_last = t_start
while surr_completed < NUM_SURR:

    # retrieve the next result
    slami_ar, slami_w1, slami_f = resq.get()
    slam_ar[surr_completed, :] = slami_ar
    slam_w1[surr_completed, :] = slami_w1
    slam_f[surr_completed, :] = slami_f        
    surr_completed += 1
        
    # predict time to go
    t_now = datetime.now()
        
    # print progress
    if (t_now - t_last).total_seconds() > 600:
        t_last = t_now
        dt = (t_now - t_start) / surr_completed * surr_todo
        log("PROGRESS: %d/%d complete, predicted completion at %s."
            % (surr_completed, NUM_SURR, str(t_now + dt)))

# wait for all workers to finish
for w in workers:
    w.join()
        
log("DONE after %s" % str(datetime.now() - t_start))

log("Saving computed spectra ...")
if USE_SURROGATE_MODEL:
    fname = 'results/%s_var_surr_comp_count_cosweights%s.bin' % (DATA_NAME, SUFFIX)
    with open(fname, 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1,
                       'slam_f' : slam_f, 'orders' : sgf.model_orders()}, f)
    log("Saved results to file %s" % fname)
elif USE_MUVAR:
    fname = 'results/%s_var_muvar_comp_count_cosweights%s.bin' % (DATA_NAME, SUFFIX)
    with open(fname, 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1,
                       'slam_f' : slam_f, 'orders' : sgf.model_orders()}, f)
    log("Saved results to file %s" % fname)
else:
    fname = 'results/%s_var_data_comp_count_cosweights%s.bin' % (DATA_NAME, SUFFIX)
    with open(fname, 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1,
                       'slam_f' : slam_f, 'orders' : sgf.model_orders()}, f)
        log("Saved results to file %s" % fname)


# show standard multiple-comparison tests assuming number of hypotheses equal to len(pvals)
pvals = compute_eigvals_pvalues(dlam, slam_ar)
log("Bonferroni correction: %d significant components." % np.sum(bonferroni_test(pvals, 0.05, NUM_SURR)))
log("Bonferroni-Holm correction: %d significant components." % np.sum(holm_test(pvals, 0.05, NUM_SURR)))
fdr_comps = np.sum(fdr_test(pvals, 0.05, NUM_SURR))
log("FDR correction: %d significant components." % fdr_comps)

log("Rendering eigenvalues from data and from surrogates")
f = plt.figure(figsize = (12,8))
FROM_EIG = 1
TO_EIG = fdr_comps + 5
x = np.arange(TO_EIG - FROM_EIG) + FROM_EIG
plt.plot(x, dlam[x-1], 'ro-')
plt.errorbar(x, np.mean(slam_ar[:, x-1], axis = 0), np.std(slam_ar[:, x-1] * 3, axis = 0), fmt = 'bo-')
plt.errorbar(x, np.mean(slam_w1[:, x-1], axis = 0), np.std(slam_w1[:, x-1] * 3, axis = 0), fmt = 'go-')
plt.errorbar(x, np.mean(slam_f[:, x-1], axis = 0), np.std(slam_f[:, x-1] * 3, axis = 0), fmt = 'ko-')
plt.legend(('Data', 'AR', 'WN', 'F1'))
plt.title('Eigenvalues for data and surrogates')
plt.savefig('%s_eigvals_comparison%s.png' % (DATA_NAME, SUFFIX))
plt.close(f)

# end of processing
log_file.close()
