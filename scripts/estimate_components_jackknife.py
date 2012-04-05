from datetime import date, datetime
from geo_field import GeoField
from multiprocessing import Pool
from component_analysis import orthomax, pca_components_gf
from geo_rendering import render_component_set
from munkres import Munkres
from spatial_model_generator import constructVAR, make_model_geofield

import math
import os.path
import sys
import numpy as np
import pylab as pb
import cPickle

#
# Current simulation parameters
#
NUM_COMPONENTS = 4
BULK_STEP = 50
POOL_SIZE = None
DISCARD_RATE = 0.1
LNO_PAR = 2


def compute_lno_sample_components(x):
    gf, Urd, i, j = x
    b = gf.data()
    b = np.vstack([b[:i,...], b[j:,...]])
    U, _, _ = pca_components_gf(b)
    Ur, _, _ = orthomax(U[:, :NUM_COMPONENTS])
    
    # compute closeness of components
    C = np.dot(Ur.T, Urd)
    
    # find optimal matching of components
    m = Munkres()
    match = m.compute(1.0 - np.abs(C))
    perm = np.zeros((NUM_COMPONENTS,), dtype = np.int)
    for i in range(len(match)):
        m_i = match[i]
        perm[m_i[0]] = m_i[1]
        # flip the sign in the matched boostrap component if the correlation was negative
        Ur[m_i[1]] = - Ur[m_i[1]] if C[m_i[0], m_i[1]] < 0.0 else Ur[m_i[1]]
        
    # reorder the bootstrap components according to the best matching 
    Ur = Ur[:, perm]
    
    return Ur


def render_set_par(x):
    render_component_set(*x)


# load up the monthly SLP geo-field
gf = GeoField()
gf.load("data/pres.mon.mean.nc", 'pres')
gf.transform_to_anomalies()
gf.normalize_monthly_variance()
gf.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))    # years 1948-2012
#gf.slice_spatial(None, [20, 87])                           # northern hemisphere, extratropical
gf.slice_spatial(None, [-88, 88])
#gf.slice_months([12, 1, 2])

#S = np.zeros(shape = (5, 10), dtype = np.int32)
#S[1:4, 0:2] = 1
#S[0:3, 6:9] = 2
#v, Sr = constructVAR(S, [0.0, 0.191, 0.120], [-0.1, 0.1], [0.00, 0.00], [0.01, 0.01])
#ts = v.simulate(768)
#gf = make_model_geofield(S, ts)


# initialize a parallel pool
pool = Pool(POOL_SIZE)

# compute components for data
Ud, sd, Vtd = pca_components_gf(gf.data())
Ud = Ud[:, :NUM_COMPONENTS]
Ur, _, its = orthomax(Ud)
print("Finished after %d iterations." % its)

t_start = datetime.now()

LNO_COUNT = len(gf.tm) // LNO_PAR
#LNO_COUNT = 4
print("Running leave one out analysis [%d samples] at %s" % (LNO_COUNT, str(t_start)))

# initialize maximal and minimal boostraps
EXTREMA_MEMORY = math.ceil(DISCARD_RATE * LNO_COUNT)
max_comp = np.tile(np.abs(Ur.copy()), (EXTREMA_MEMORY + BULK_STEP, 1, 1))
min_comp = np.tile(np.abs(Ur.copy()), (EXTREMA_MEMORY + BULK_STEP, 1, 1))
mean_comp = np.zeros_like(Ur)
var_comp = np.zeros_like(Ur)
    
bsmp_done = 0

print("Running parallel generation of %d samples." % LNO_COUNT)
while bsmp_done < LNO_COUNT:
    
    t1 = datetime.now()

    bsmp_todo_now = min(LNO_COUNT - bsmp_done, BULK_STEP)
    
    # construct the surrogates in parallel
    # we can duplicate the list here without worry as it will be copied into new python processes
    # thus creating separate copies of sd
    slam_list = pool.map(compute_lno_sample_components,
                         [(gf, Ur, i*LNO_PAR, (i+1)*LNO_PAR) for i in range(bsmp_done, bsmp_done+bsmp_todo_now)])
    
    t1a = datetime.now()

    # third sorting strategy (for large bootstrap samples) - insert in everything that was in this step
    # and sort only once
    for i, Urb in zip(range(len(slam_list)), slam_list):
        min_comp[EXTREMA_MEMORY+i, :, :] = np.abs(Urb)
        max_comp[BULK_STEP-i-1, :, :] = np.abs(Urb)
        
        bsmp_done += 1
        delta = Urb - mean_comp
        mean_comp += delta / bsmp_done
        var_comp += delta * (Urb - mean_comp)
        
    # sort the entries along first axis
    min_comp[:EXTREMA_MEMORY + bsmp_todo_now, :, :].sort(axis = 0)
    max_comp[BULK_STEP - bsmp_todo_now:, :, :].sort(axis = 0)
    
    # print progress based on rate of completion from beginning of computation (from t_start)
    t2 = datetime.now()
    dt = (t2 - t_start) * (LNO_COUNT - bsmp_done) // bsmp_done
    
    da = t2 - t1a
    db = t2 - t1
        
    # print progress
    print("PROGRESS [%s]: %d/%d complete (sorting %g percent), predicted completion at %s." % 
          (str(t2), bsmp_done, LNO_COUNT,
           int((da.seconds + da.microseconds/10e6) / (db.seconds + db.microseconds/10e6) * 100),
           str(t2 + dt)))

var_comp /= (bsmp_done - 1)
    
print("DONE at %s after %s" % (str(datetime.now()), str(datetime.now() - t_start)))


# reshape all the fields back into correct spatial dimensions to match lon/lat of the original geofield
max_comp = gf.reshape_flat_field(max_comp[BULK_STEP, :, :])
min_comp = gf.reshape_flat_field(min_comp[EXTREMA_MEMORY, :, :])
mean_comp = gf.reshape_flat_field(mean_comp)
var_comp = gf.reshape_flat_field(var_comp)

# Ud and Ur are reshaped into new variables as the old ones may be reused in a new computation
Ud_gf = gf.reshape_flat_field(Ud)
Ur_gf = gf.reshape_flat_field(Ur)


# save the results to file
with open('results/lno05_results.bin', 'w') as f:
    cPickle.dump({ 'Ud' : Ud_gf, 'Ur' : Ur_gf, 'max' : max_comp, 'min' : min_comp,
                  'mean' : mean_comp, 'var' : var_comp}, f)

t_start = datetime.now()
#print("Rendering components in parallel at [%s] ..." % str(t_start))
#render_list_triples = [ (Ur_gf[i, ...], min_comp[i, ...], max_comp[i, ...], 
#                        ['data', 'min', 'max'], 
#                        gf.lats, gf.lons, True, 'figs/nhemi_comp%02d_varimax_rough.png' % (i+1),
#                        'Component %d' % (i+1)) for i in range(NUM_COMPONENTS)]
#
#pool.map(render_triples_par, render_list_triples)
#
#print("DONE after [%s]." % (datetime.now() - t_start))


render_list = [ ([ mean_comp[i, ...], Ur_gf[i, ...], mean_comp[i, ...] / (var_comp[i,...] ** 0.5 + 0.01)], 
                [ 'mean', 'data', 'T'],
                gf.lats, gf.lons, False, 'figs/nhemi_comp%02d_varimax_mn.png' % (i+1),
                'Component %d' % (i+1)) for i in range(NUM_COMPONENTS)]

map(render_set_par, render_list)

print("DONE after [%s]." % (datetime.now() - t_start))

