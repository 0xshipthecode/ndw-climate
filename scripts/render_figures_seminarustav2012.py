

from datetime import date
from geo_field import GeoField
from var_model import VARModel
from geo_rendering import render_component_single
import matplotlib.pyplot as plt

import cPickle
import numpy as np


def render_synth_model_component_eigvals():
    
    with open('results/slp_eigvals_three_surrogates_synth_model.bin', 'r') as f:
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
    
    plt.figure(figsize = (10,8))
    
    plt.title('Synthetic model eigenvalues from data and surrogates')    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
    mn = np.mean(slam_w2, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.3, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.axes([0.3, 0.5, 0.5, 0.3], axisbg = 'w')
    plt.axis([0, 10, 9, 23])
    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
    mn = np.mean(slam_w2, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_w2[p0025, :], slam_w2[p0975, :] - mn]), fmt = 'ko-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.3, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
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

    
    
if __name__ == '__main__':
    
    render_synth_model_component_eigvals()
#    render_model_order_synth_model()

