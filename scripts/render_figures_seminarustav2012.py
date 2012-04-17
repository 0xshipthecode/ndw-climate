

from datetime import date, datetime
from geo_field import GeoField
from geo_rendering import render_component_set
from var_model import VARModel
from geo_rendering import render_component_single
from error_metrics import estimate_snr, marpe_error, mse_error

import matplotlib.pyplot as plt
from multiprocessing import Pool
from component_analysis import matched_components, match_components_munkres
import cPickle
import numpy as np


def render_synth_surrogate_model_component_eigvals():
    
    with open('results/synth_model_surrogate_comp_count_multi_surrogates_190smp.bin', 'r') as f:
        res = cPickle.load(f)

    dlam = res['dlam']
    slam_ar = res['slam_ar']
    slam_w1 = res['slam_w1']
#    slam_w2 = res['slam_w2']
    slam_f = res['slam_f']
    
    slam_ar.sort(axis = 0)
    slam_w1.sort(axis = 0)
#    slam_w2.sort(axis = 0)
    slam_f.sort(axis = 0)
    
    N = len(dlam)
    S = len(slam_ar)
    p0025 = int(S * 0.025)
    p0975 = int(S * 0.975)
    p0500 = int(S * 0.5)
    
    plt.figure(figsize = (10,8))
    
    plt.title('Surrogate synthetic model eigenvalues')
    plt.xlabel('Eigenvalue index [-]')
    plt.ylabel('Eigenvalue magnitude [-]')
   
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
#    mn = np.mean(slam_w2, axis = 0)
#    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.3, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.legend(('Surr/Data', 'AR', 'WN', 'Fourier'))
    
    plt.show()


def render_synth_model_component_eigvals():
    
    with open('results/synth_model_comp_count_multi_surrogates_190smp.bin', 'r') as f:
        res = cPickle.load(f)
        
    dlam = res['dlam']
    slam_ar = res['slam_ar']
    slam_w1 = res['slam_w1']
#    slam_w2 = res['slam_w2']
    slam_f = res['slam_f']
    
    slam_ar.sort(axis = 0)
    slam_w1.sort(axis = 0)
#    slam_w2.sort(axis = 0)
    slam_f.sort(axis = 0)
    
    N = len(dlam)
    S = len(slam_ar)
    p0025 = int(S * 0.025)
    p0975 = int(S * 0.975)
    p0500 = int(S * 0.5)
    
    plt.figure()
    
    plt.title('Synthetic model eigenvalues from data and surrogates')    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
#    mn = np.mean(slam_w2, axis = 0)
#    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.legend(('Data', 'AR', 'WN', 'Fourier'))
    
    plt.axes([0.3, 0.3, 0.5, 0.3], axisbg = 'w')
    plt.axis([2, 6, 11, 15])
    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
#    mn = np.mean(slam_w2, axis = 0)
#    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.show()
    
    
def render_model_order_synth_model():
    
    with open('results/slp_eigvals_three_surrogates_synth_model.bin', 'r') as f:
        res = cPickle.load(f)
        
    print res.keys()
    mo = res['orders']
    
    plt.figure(figsize = (10,6))
    plt.imshow(mo, interpolation = 'nearest')
    plt.title('Model order of null AR model', fontsize = 16)
    plt.colorbar()
    
    plt.show()
    
    
def render_model_order_slp():
    
    with open('results/slp_eigvals_multi_surrogates_nh_var.bin', 'r') as f:
        res = cPickle.load(f)

    

def render_slp_nh_component_eigvals():
    
    with open('results/slp_eigvals_multi_surrogates_nh_var.bin', 'r') as f:
        res = cPickle.load(f)
        
    dlam = res['dlam']
    slam_ar = res['slam_ar']
    slam_w1 = res['slam_w1']
#    slam_w2 = res['slam_w2']
    slam_f = res['slam_f']
    
    slam_ar.sort(axis = 0)
    slam_w1.sort(axis = 0)
#    slam_w2.sort(axis = 0)
    slam_f.sort(axis = 0)
    
    N = len(dlam)
    S = len(slam_ar)
    p0025 = int(S * 0.025)
    p0975 = int(S * 0.975)
    p0500 = int(S * 0.5)
    
    plt.figure(figsize = (10,8))
    
    plt.title('Synthetic model eigenvalues from data and surrogates')    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
#    mn = np.mean(slam_w2, axis = 0)
#    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.axes([0.3, 0.5, 0.5, 0.3], axisbg = 'w')
    plt.axis([30, 50, 5, 30])
    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
#    mn = np.mean(slam_w2, axis = 0)
#    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.show()
    
    
def render_slp_components():
    
    with open('results/slp_nh_var_bootstrap_results_b1000.bin', 'r') as f:
        d = cPickle.load(f)
        
    mn = d['mean']
    
    mn_mask = (np.abs(mn) > 1.0 / mn.shape[0]**0.5)
    mn_thr = mn * mn_mask
    print np.sum(np.sum(mn_mask, axis= 1) == 0)
    cid = np.argmax(np.abs(mn) * mn_mask, axis = 1)[:, np.newaxis] + 1
    
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.slice_spatial(None, [20, 89])
    mnd = gf.reshape_flat_field(cid)
    
#    plt.figure()
#    plt.hist(cid, bins = 43)
#    plt.show()
    
    f = render_component_single(mnd[0, :, :], gf.lats, gf.lons, False, None,
                                'NH Extratropical Components',
                                cmap = plt.get_cmap('gist_ncar'))
    plt.show()
    
    
def render_slp_component_element_values():
    
    with open('results/slp_nh_var_bootstrap_results_b1000_cosweights.bin', 'r') as f:
        d = cPickle.load(f)
        
    mn = d['mean']
    
    cid = np.amax(np.abs(mn), axis = 1)[:, np.newaxis]
    
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.slice_spatial(None, [20, 89])
    mnd = gf.reshape_flat_field(cid)
    
    f = render_component_single(mnd[0, :, :], gf.lats, gf.lons, False, None,
                                'NH Extratropical Components - max values')
    plt.show()


def plot_slp_components_stability_b1000():
    
    with open('results/slp_nh_var_bootstrap_results_b1000.bin', 'r') as f:
        d = cPickle.load(f)
        
    mn = d['mean']

    with open('results/slp_nh_var_bootstrap_results_b1000_2ndrun.bin', 'r') as f:
        d = cPickle.load(f)
        
    mn2 = d['mean']

    with open('results/slp_nh_var_surrogate_bootstrap_results_b1000.bin', 'r') as f:
        d = cPickle.load(f)
        
    mns = d['mean']
    
    plt.figure()
    plt.plot(np.arange(mn.shape[1]) + 1, np.sum(mn**2, axis = 0), 'ro-')
    plt.plot(np.arange(mn2.shape[1]) + 1, np.sum(mn2**2, axis = 0), 'bo-')
    plt.plot(np.arange(mns.shape[1]) + 1, np.sum(mns**2, axis = 0), 'go-')
    plt.title('Squared 2-norm of vectors')
    plt.legend(('B1000, run 1', 'B1000, run 2', 'Surrogate'))
    plt.show()
    
    
def plot_slp_model_orders():
    
    with open('results/slp_eigvals_multi_surrogates_nh_var.bin', 'r') as f:
        d = cPickle.load(f)
        
    print(d.keys())
    o = d['orders'][:, np.newaxis]
    print(o.shape)
    
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.slice_spatial(None, [20, 89])
#    od = gf.reshape_flat_field(o)
    
    f = render_component_single(o[:, 0, :], gf.lats, gf.lons, False, None,
                                'NH Extratropical Components - AR model order')
    plt.show()
    
    
def plot_slp_component_eigvals():
    
    with open('results/slp_eigvals_multi_surrogates_nh_var.bin', 'r') as f:
        res = cPickle.load(f)
        
    dlam = res['dlam']
    slam_ar = res['slam_ar']
    slam_w1 = res['slam_w1']
    slam_w2 = res['slam_w2']
    slam_f = res['slam_f']
    
    slam_ar.sort(axis = 0)
    slam_w1.sort(axis = 0)
    slam_w2.sort(axis = 0)
    slam_f.sort(axis = 0)
    
    N = len(dlam)
    S = len(slam_ar)
    p0025 = int(S * 0.025)
    p0975 = int(S * 0.975)
    p0500 = int(S * 0.5)
    
    plt.figure()
    
    plt.title('SLP NH Extratropical - Eigvals from data and surrogates')    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
#    mn = np.mean(slam_w2, axis = 0)
#    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.xlabel('Eigenvalue index [-]')
    plt.ylabel('Eigenvalue magnitude [-]')
    
    plt.legend(('Data', 'AR', 'WN', 'Fourier'))
    
    plt.axes([0.3, 0.3, 0.5, 0.3], axisbg = 'w')
    plt.axis([42, 48, 7, 12])
    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
#    mn = np.mean(slam_w2, axis = 0)
#    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.show()


def render_set_stub(x):
    render_component_set(*x)


def plot_components_slp():
    
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.slice_spatial(None, [20, 89])
    
    with open('results/slp_nh_var_bootstrap_results_b1000_cosweights.bin', 'r') as f:
        d = cPickle.load(f)

    mn = d['mean']
    
    mn_gf = gf.reshape_flat_field(mn)
    
    t_start = datetime.now()
    thr = 1.0 / mn.shape[0]**0.5

    pool = Pool(4)
    render_list_triples = [ ([mn_gf[i, ...], 
                             mn_gf[i, ...] * (np.abs(mn_gf[i,...]) > thr)],
                             [ 'Mean', 'Mean:Thr'],
                             gf.lats, gf.lons, False,
#                             'figs/nhemi_comp%02d_varimax_mn.png' % (i+1),
                             'figs/nhemi_comp%02d_varimax_mn.pdf' % (i+1),
                             'Component %d' % (i+1))
                                for i in range(mn.shape[1])]

    # use less nodes to render the maps due to memory constraints
    pool.map(render_set_stub, render_list_triples)
    pool.close()

    # clean up memory
    del render_list_triples

    print("DONE after [%s]." % (datetime.now() - t_start))
    
    
def synth_model_plot_component_matching():

    with open('results/synth_model_component_robustness_190smp_4comps.bin', 'r') as f:
        d = cPickle.load(f)
        
    Uopt = d['Uopt']
    Ur = d['Ur']
    mean_comp = d['mean']
    var_comp = d['var']
    
    perm, sf = match_components_munkres(Uopt, Ur)
    perm = perm[:3]
    sf = sf[:, :3]
    rst = np.setdiff1d(np.arange(Ur.shape[1]), np.sort(perm[:3]), False)
    print rst
    Urm = np.hstack([Ur[:, perm] * sf, Ur[:, rst]])
    print Urm.shape
    
    perm, sf = match_components_munkres(Uopt, mean_comp)
    perm = perm[:3]
    sf = sf[:, :3]
    rst = np.setdiff1d(np.arange(mean_comp.shape[1]), np.sort(perm[:3]), False)
    print rst
    Umvm = np.hstack([mean_comp[:, perm] * sf, mean_comp[:, rst]])
    Umvv = np.hstack([var_comp[:, perm], var_comp[:, rst]]) 
    
    print Umvm.shape
    print Umvv.shape
    
    p2 = list(perm)
    for i in range(mean_comp.shape[1]):
        if not i in p2:
            p2.append(i)
    
    print estimate_snr(Uopt, Urm[:, :3])
    print estimate_snr(Uopt, Umvm[:, :3])
    
    print mse_error(Uopt, Urm[:, :3])
    print mse_error(Uopt, Umvm[:, :3])
    
    print marpe_error(Uopt, Urm[:, :3])
    print marpe_error(Uopt, Umvm[:, :3])
    
    print np.sum(Umvv, axis = 0)
    print np.sum(Umvm ** 2, axis = 0)
    print np.sum(Umvm ** 2, axis = 0) / np.sum(Umvv[:, p2], axis = 0)
    
    plt.figure(figsize = (12, 16))
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
    for i in range(4):
        
        plt.subplot(4, 3, i*3+1)
        if(i < 3):
            plt.plot(Uopt[:,i], 'r-')
        plt.plot(Umvm[:,i], 'b-')
        plt.plot(np.ones((Uopt.shape[0],1)) * (1.0 / Uopt.shape[0]**0.5), 'g-', linewidth = 2)
        plt.title('Component %d [bootstrap mean]' % (i+1))
        
        plt.subplot(4, 3, i*3+2)
        if(i < 3):
            plt.plot(Uopt[:,i], 'r-')
        plt.plot(Urm[:,i], 'b-')
        plt.plot(np.ones((Uopt.shape[0],1)) * (1.0 / Uopt.shape[0]**0.5), 'g-', linewidth = 2)
        plt.title('Component %d [data]' % (i+1))
    
        plt.subplot(4, 3, i*3+3)
        plt.plot(Umvv[:,i], 'b-')
        plt.title('Component %d [variance]' % (i+1))
        
    plt.show()

    
if __name__ == '__main__':
    
#    render_slp_nh_component_eigvals()
#    plot_components_slp()
    render_slp_component_element_values()
    
#    render_synth_model_component_eigvals()
#    render_synth_surrogate_model_component_eigvals()
    
#    render_model_order_synth_model()
#    render_slp_components()
#    plot_slp_components_stability_b1000()
#    plot_slp_model_orders()
#    plot_slp_component_eigvals()
#    synth_model_plot_component_matching()