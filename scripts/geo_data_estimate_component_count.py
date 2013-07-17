# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib
matplotlib.use('Agg')

from datetime import date, datetime
from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from multiprocessing import Pool
from component_analysis import pca_eigvals_gf, pca_components_gf, matched_components, orthomax
from spatial_model_generator import constructVAR, make_model_geofield
from geo_data_loader import load_monthly_data_general, load_monthly_sat_all
from geo_rendering import render_component_single
from multi_stats import compute_eigvals_pvalues, fdr_test, bonferroni_test

import os.path
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import scipy.signal as sps

# <codecell>

#
# Current simulation parameters
#
NUM_SURR = 20000
SURR_REPORT_STEP = 200
USE_MUVAR = False
USE_SURROGATE_MODEL = False
COSINE_REWEIGHTING = True
NUM_EIGVALS = 100
POOL_SIZE = 20
MAX_AR_ORDER = 30
RECOMPUTE_MODEL = True
DETREND = True
DATA_NAME = 'sat_all'
SUFFIX ="_detrended"


loader_functions = {
    'sat_all' : load_monthly_sat_all,
    'slp_all' : load_monthly_slp_all
}

# <codecell>

# initialize random number generators
#random.seed()
np.random.seed()

# <markdowncell>

# **Loading & filtering phase**
# 
# Now we must filter the data in frequency, the data loading has thus been moved here.

# <codecell>

os.chdir('/home/martin/Projects/Climate/ndw-climate/')

print("[%s] Loading geo field..." % (str(datetime.now())))

#gf = load_monthly_data_general("data/hgt.mon.mean.nc", "hgt",
#                               date(1948, 1, 1), date(2012, 1, 1),
#                               None, None, None, 5)

gf = loader_functions[DATA_NAME]()

# if detrend is required, do it now
if DETREND:
    gf.detrend()


# load up the monthly SLP geo-field
if USE_MUVAR:
    print("[%s] Constructing F-2 surrogate ..." % (str(datetime.now())))
    sgf = SurrGeoFieldAR()
    sgf.copy_field(gf)
    sgf.construct_fourier2_surrogates()
    sgf.d = sgf.sd.copy()

    # slide in fourier2 surrogate
    orig_gf = gf
    gf = sgf

# load up the monthly SLP geo-field
print("[%s] Done loading, data hase shape %s." % (str(datetime.now()), gf.d.shape))

# <markdowncell>

# **Surrogate construction**
# 
# The next cell constructs surrogates in three ways.  At this point I'm verifying the previous results obtained
# using residuals by feeding in normally distributed noise according to the model covariance matrix (noise variance in case of AR(k) models).

# <codecell>

def compute_surrogate_cov_eigvals(sd):
    
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
    
    return sm_ar, sm_w1, sm_f

# <markdowncell>

# The above cell also returns "matched" components for the null model.

# <codecell>

print("[%s] Constructing surrogate model ..." % (str(datetime.now())))
pool = Pool(POOL_SIZE)
#pool = None
sgf = SurrGeoFieldAR([0, MAX_AR_ORDER], 'sbc')
sgf.copy_field(gf)
sgf.prepare_surrogates(pool)
sgf.construct_surrogate_with_noise()
if pool is not None:
    pool.close()
    pool.join()
    del pool
print("[%s] Constructed."  % (str(datetime.now())))

if USE_SURROGATE_MODEL:
    # HACK to replace original data with surrogates
    gf.d = sgf.sd.copy()
    sgf.d = sgf.sd.copy()
    print("Replaced synth model with surrogate model to check false positives.")

# analyze data & obtain eigvals and surrogates
print("[%s] Analyzing data ..." % (str(datetime.now())))
d = gf.data()
if COSINE_REWEIGHTING:
    d *= gf.qea_latitude_weights()
dlam = pca_eigvals_gf(d)[:NUM_EIGVALS]
    
print("[%s] Data analysis DONE." % (str(datetime.now())))

# <markdowncell>

# **Show the variance of the data (filtered)**

# <markdowncell>

# **Show a plot of the model orders**

# <codecell>

mo = sgf.model_orders()
render_component_single(mo, gf.lats, gf.lons, plt_name = 'Model orders of AR surrogates', fname='ar_model_order.png')

# <codecell>

pool = Pool(POOL_SIZE)
log = open('geodata_estimate_component_count-%s.log' % datetime.now().strftime('%Y%m%d-%H%M'), 'w')
log.write('Analyzing data %s\n' % DATA_NAME)

# storage for three types of surrogates
slam_ar = np.zeros((NUM_SURR, NUM_EIGVALS))
slam_w1 = np.zeros((NUM_SURR, NUM_EIGVALS))
slam_f = np.zeros((NUM_SURR, NUM_EIGVALS))

surr_completed = 0

# construct the job queue
job_list = []
job_list.extend([sgf] * NUM_SURR)

TOTAL_JOBS = len(job_list)
print("[%s] I have %d jobs in queue." % (str(datetime.now()), TOTAL_JOBS))
log.write("[%s] I have %d jobs in queue.\n" % (str(datetime.now()), TOTAL_JOBS))

# generate and compute eigenvalues for 20000 surrogates
t_start = datetime.now()
    
# construct the surrogates in parallel
# we can duplicate the list here without worry as it will be copied into new python processes
# thus creating separate copies of sd
print("[%s] Running parallel generation of surrogates and analysis." % str(t_start))
log.write("[%s] Running parallel generation of surrogates and analysis.\n" % str(t_start))

while len(job_list) > 0:

    t1 = datetime.now()
    
    surr_todo = min(SURR_REPORT_STEP, len(job_list))
    
    # take first surr_todo jobs from the job_list
    job_donow_list = job_list[:surr_todo]
    job_list = job_list[surr_todo:]
    
    # run computations for all jobs
    slam_list = pool.map(compute_surrogate_cov_eigvals, job_donow_list)

    # rearrange into numpy array (can I use vstack for this?)
    for i in range(len(slam_list)):
        slami_ar, slami_w1, slami_f = slam_list[i]
        slam_ar[surr_completed, :] = slami_ar
        slam_w1[surr_completed, :] = slami_w1
        slam_f[surr_completed, :] = slami_f        

        surr_completed += 1
        
    # predict time to go
    t2 = datetime.now()
    dt = (t2 - t1) * (TOTAL_JOBS - surr_completed) // surr_todo
        
    # print progress
    print("PROGRESS [%s]: %d/%d complete, predicted completion at %s." % 
          (str(t2), surr_completed, TOTAL_JOBS, str(t2 + dt)))
    log.write("PROGRESS [%s]: %d/%d complete, predicted completion at %s.\n" % 
          (str(t2), surr_completed, TOTAL_JOBS, str(t2 + dt)))
    log.flush()
        
pool.close()
pool.join()

print("DONE at %s after %s" % (str(datetime.now()), str(datetime.now() - t_start)))
log.write("DONE at %s after %s\n" % (str(datetime.now()), str(datetime.now() - t_start)))
log.close()

# <codecell>

print("Saving computed spectra ...")
# save the results to file
if USE_SURROGATE_MODEL:
    with open('results/%s_var_surrogate_comp_count_cosweights%s.bin' % (DATA_NAME, SUFFIX), 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1, 'slam_f' : slam_f,
                      'orders' : sgf.model_orders()}, f)
elif USE_MUVAR:
    with open('results/%s_var_muvar_comp_count_cosweights%s.bin' % (DATA_NAME, SUFFIX), 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1, 'slam_f' : slam_f,
                      'orders' : sgf.model_orders()}, f)
else:
    with open('results/%s_var_data_comp_count_cosweights%s.bin' % (DATA_NAME, SUFFIX), 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1, 'slam_f' : slam_f,
                      'orders' : sgf.model_orders()}, f)

# <codecell>

f = plt.figure(figsize = (10,6))
FROM_EIG = 1
TO_EIG = 50
x = np.arange(TO_EIG - FROM_EIG) + FROM_EIG
plt.plot(x, dlam[x-1], 'ro-')
plt.errorbar(x, np.mean(slam_ar[:, x-1], axis = 0), np.std(slam_ar[:, x-1] * 3, axis = 0), fmt = 'bo-')
plt.errorbar(x, np.mean(slam_w1[:, x-1], axis = 0), np.std(slam_w1[:, x-1] * 3, axis = 0), fmt = 'go-')
plt.errorbar(x, np.mean(slam_f[:, x-1], axis = 0), np.std(slam_f[:, x-1] * 3, axis = 0), fmt = 'ko-')
plt.legend(('Data', 'AR', 'WN', 'F1'))
plt.title('Eigenvalues for data and surrogates')
plt.savefig('eigvals_comparison.png')
plt.close(f)

# <codecell>

print slam_ar.shape[0]
pvals = compute_eigvals_pvalues(dlam, slam_ar)
print("Bonferroni correction: %d significant components." % np.sum(bonferroni_test(pvals, 0.05, slam_ar.shape[0])))
print("FDR correction: %d significant components." % np.sum(fdr_test(pvals, 0.05, slam_ar.shape[0])))
