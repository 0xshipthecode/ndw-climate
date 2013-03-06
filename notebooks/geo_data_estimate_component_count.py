# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from datetime import date, datetime
from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
from multiprocessing import Pool
from component_analysis import pca_eigvals_gf, pca_components_gf, matched_components, orthomax
from spatial_model_generator import constructVAR, make_model_geofield
from scipy.signal import detrend, iirdesign, filtfilt, lfilter, freqz
from geo_data_loader import load_monthly_slp_all, load_monthly_sat_all, load_monthly_data_general
from geo_rendering import render_component_single
from multi_stats import compute_eigvals_pvalues, fdr_test, bonferroni_test

import os.path
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import scipy.signal as sps

# <codecell>

# filtering routine for the data
def apply_filter(d, b, a):
    for i in range(d.shape[1]):
        for j in range(d.shape[2]):
            d[:, i,j] = sps.filtfilt(b, a, d[:, i, j])

# <codecell>

#
# Current simulation parameters
#
NUM_SURR = 100
SURR_REPORT_STEP = 50
USE_SURROGATE_MODEL = False
COSINE_REWEIGHTING = True
NUM_EIGVALS = 100
POOL_SIZE = 20
MAX_AR_ORDER = 30
RECOMPUTE_MODEL = True
PERIOD_RANGES = [ (18, 22), (9, 14), (6.5, 8.9), (3.5, 6), (2, 3) ]
#PERIOD_RANGES = [ (18, 22) ]

# <codecell>

# construction of frequency ranges (in frequency is measured in 1/months)
Fs = 12.0  # (months==samples)/year
Nqs = Fs / 2.0
frequency_ranges = [ ( 1.0 / ph / Nqs, 1.0 / pl / Nqs) for (pl, ph) in PERIOD_RANGES ]

NF = len(frequency_ranges)
filters = []
for (i, (fl, fh)) in zip(range(NF), frequency_ranges):
    b, a = iirdesign(wp = [fl, fh], ws = [ fl * 0.5, fh * 1.5 ], gpass = 1.0, gstop = 20, ftype = 'ellip', output = 'ba' )
    filters.append((b, a))

# <codecell>

# initialize random number generators
random.seed()
np.random.seed()

# <markdowncell>

# **Loading & filtering phase**
# 
# Now we must filter the data in frequency, the data loading has thus been moved here.

# <codecell>

os.chdir('/home/martin/Projects/Climate/ndw-climate/')

# load up the monthly SLP geo-field
print("[%s] Loading SAT/ALL geo field..." % (str(datetime.now())))

gf = load_monthly_sat_all()

# load without variance normalization
#gf = GeoField()
#gf.load('data/air.mon.mean.nc', 'air')
#gf.slice_level(0)
#gf.transform_to_anomalies()
#gf.slice_spatial(None, [-89, 89])
#gf.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))

# load up the monthly SLP geo-field
print("[%s] Constructing F2 surrogate ..." % (str(datetime.now())))
sgf = SurrGeoFieldAR()
sgf.copy_field(gf)
sgf.construct_fourier2_surrogates()
#HACK to replace original data with surrogate
sgf.d = sgf.sd.copy()

# slide in fourier2 surrogate
orig_gf = gf
gf = sgf

#gfo = load_monthly_slp_all()
#gf = load_monthly_data_general('../data/slp.mon.mean.nc', 'slp', date(1948, 1, 1), date(2012, 1, 1), None, None, [-89, 0], None)
#gf = load_monthly_data_general('data/air.mon.mean.nc', 'air', date(1948, 1, 1), date(2012, 1, 1), None, None, [-89, 0], 0)

# load up the monthly SLP geo-field
print("[%s] Done loading." % (str(datetime.now())))

print gf.d.shape

# <markdowncell>

# **Surrogate construction**
# 
# The next cell constructs surrogates in three ways.  At this point I'm verifying the previous results obtained
# using residuals by feeding in normally distributed noise according to the model covariance matrix (noise variance in case of AR(k) models).

# <codecell>

def compute_surrogate_cov_eigvals(x):
    sd, f, (b,a) = x
    
    # construct AR/SBC surrogates
    sd.construct_surrogate_with_noise()
    d = sd.surr_data()
    if f is not None:
        apply_filter(d, b, a)
    d /= np.std(d, axis = 0)
    if COSINE_REWEIGHTING:
        d *= sd.qea_latitude_weights()
    sm_ar = pca_eigvals_gf(d)
    sm_ar = sm_ar[:NUM_EIGVALS]
    
    # construct fourier surrogates
    sd.construct_fourier1_surrogates()
    d = sd.surr_data()
    if f is not None:
        apply_filter(d, b, a)
    d /= np.std(d, axis = 0)
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
    if f is not None:
        apply_filter(d, b, a)
    d /= np.std(d, axis = 0)
    if COSINE_REWEIGHTING:
        d = d * sd.qea_latitude_weights()
    sm_w1 = pca_eigvals_gf(d)
    sm_w1 = sm_w1[:NUM_EIGVALS]
    
    return sm_ar, sm_w1, sm_f, f

# <markdowncell>

# The above cell also returns "matched" components for the null model.

# <codecell>

print("[%s] Constructing surrogate model ..." % (str(datetime.now())))
pool = Pool(POOL_SIZE)
sgf = SurrGeoFieldAR([0, MAX_AR_ORDER], 'sbc')
sgf.copy_field(gf)
sgf.prepare_surrogates(pool)
sgf.construct_surrogate_with_noise()
pool.close()
del pool
print("[%s] Constructed."  % (str(datetime.now())))

if USE_SURROGATE_MODEL:
    # HACK to replace original data with surrogates
    gf.d = sgf.sd.copy()
    sgf.d = sgf.sd.copy()
    print("Replaced synth model with surrogate model to check false positives.")

dlam = np.zeros((NUM_EIGVALS, NF))
    
# analyze data & obtain eigvals and surrogates
for f in range(NF):
    print("[%s] Analyzing data (period %g-%g yrs) ..." % (str(datetime.now()), PERIOD_RANGES[f][0], PERIOD_RANGES[f][1]))
    d = gf.data()
    b, a = filters[f]
    apply_filter(d, b, a)
    d /= np.std(d, axis = 0)
    if COSINE_REWEIGHTING:
        d *= gf.qea_latitude_weights()
    dlam[:, f] = pca_eigvals_gf(d)[:NUM_EIGVALS]
    print("[%s] Frequency-specific analysis completed." % (str(datetime.now())))
    
print("[%s] Data analysis DONE." % (str(datetime.now())))

# <markdowncell>

# **Show the variance of the data (filtered)**

# <markdowncell>

# **Show a plot of the model orders**

# <codecell>

mo = sgf.model_orders()
plt = render_component_single(mo, gf.lats, gf.lons, plt_name = 'Model orders of AR surrogates')

# <codecell>

pool = Pool(POOL_SIZE)
log = open('geodata_estimate_component_count-%s.log' % datetime.now().strftime('%Y%m%d-%H%M'), 'w')

# storage for three types of surrogates
slam_ar = np.zeros((NUM_SURR, NUM_EIGVALS, NF))
slam_w1 = np.zeros((NUM_SURR, NUM_EIGVALS, NF))
#slam_w2 = np.zeros((NUM_SURR, NUM_EIGVALS))
slam_f = np.zeros((NUM_SURR, NUM_EIGVALS, NF))

surr_completed = 0
surr_completed_f = np.zeros((NF,))

# construct the job queue
job_list = []
for f in range(NF):
    job_list.extend([(sgf, f, filters[f])] * NUM_SURR)

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
    
    # get the results
    slam_list = pool.map(compute_surrogate_cov_eigvals, job_donow_list)
    
    # rearrange into numpy array (can I use vstack for this?)
    for i in range(len(slam_list)):
        slami_ar, slami_w1, slami_f, f = slam_list[i]
        slam_ar[surr_completed_f[f], :, f] = slami_ar
        slam_w1[surr_completed_f[f], :, f] = slami_w1
        slam_f[surr_completed_f[f], :, f] = slami_f
        
        # robust computation of mean over surrogates
        surr_completed += 1
        surr_completed_f[f] += 1
        
        
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

print("DONE at %s after %s" % (str(datetime.now()), str(datetime.now() - t_start)))
log.write("DONE at %s after %s\n" % (str(datetime.now()), str(datetime.now() - t_start)))
log.close()

# <codecell>

print("Saving computed spectra ...")
# save the results to file
if USE_SURROGATE_MODEL:
    with open('results/sat_all_freq_surrogate_comp_count_cosweights_pilot.bin', 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1, 'slam_f' : slam_f,
                      'orders' : sgf.model_orders(), 'period_ranges' : PERIOD_RANGES}, f)
else:
    with open('results/sat_all_var_muvar_freq_comp_count_cosweights_pilot.bin', 'w') as f:
        cPickle.dump({ 'dlam' : dlam, 'slam_ar' : slam_ar, 'slam_w1' : slam_w1, 'slam_f' : slam_f,
                      'orders' : sgf.model_orders(), 'period_ranges' : PERIOD_RANGES}, f)

# <codecell>

figure(figsize = (10,6))
FROM_EIG = 1
TO_EIG = 10
FR = 0
x = np.arange(TO_EIG - FROM_EIG) + FROM_EIG
plot(x, dlam[x-1,FR], 'ro-')
errorbar(x, np.mean(slam_ar[:, x-1,FR], axis = 0), np.std(slam_ar[:, x-1,FR] * 3, axis = 0), fmt = 'bo-')
errorbar(x, np.mean(slam_w1[:, x-1,FR], axis = 0), np.std(slam_w1[:, x-1,FR] * 3, axis = 0), fmt = 'go-')
errorbar(x, np.mean(slam_f[:, x-1,FR], axis = 0), np.std(slam_f[:, x-1,FR] * 3, axis = 0), fmt = 'ko-')
legend(('Data', 'AR', 'WN', 'F1'))
title('Eigenvalues for frequency range %d' % FR)

# <codecell>

print slam_ar.shape[0]
pvals = compute_eigvals_pvalues(dlam, slam_ar)
print("Bonferroni correction: %d significant components." % np.sum(bonferroni_test(pvals, 0.05, slam_ar.shape[0])))
print("FDR correction: %d significant components." % np.sum(fdr_test(pvals, 0.05, slam_ar.shape[0])))

