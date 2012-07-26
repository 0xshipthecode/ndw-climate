
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
from mpl_toolkits import basemap
import geo_data_loader
from scipy.signal.spectral import lombscargle


#FILE_NAME_EIGS = 'results/slp_nh_var_comp_count_cosweights.bin'

#FILE_NAME_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_unnorm_normrows.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_unnorm_normrows.bin'

#FILE_NAME_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_normrows.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_nh_var_surrogate_bootstrap_results_b1000_cosweights_normrows.bin'

#FILE_NAME_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_nh_var_surrogate_bootstrap_results_b1000_cosweights.bin'
#METHOD_NAME = 'URPCA'

#FILE_NAME_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_fastica.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_nh_var_surrogate_bootstrap_results_b1000_cosweights_fastica.bin'
#METHOD_NAME = 'FastICA'
    
#FILE_NAME_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_fasticaT.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_fasticaT.bin'
#METHOD_NAME = 'FastICAT'

#FILE_NAME_COMPS = 'results/real_slp_nh_var_bootstrap_results_b1000_cosweights.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_nh_var_surrogate_bootstrap_results_b1000_cosweights.bin'
#METHOD_NAME = 'URPCA'

#FILE_NAME_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_spca.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_nh_var_surrogate_bootstrap_results_b1000_cosweights.bin'
#METHOD_NAME = 'SPCA'

#FILE_NAME_EIGS = 'results/slp_all_var_comp_count_cosweights.bin'
#FILE_NAME_COMPS = 'results/slp_all_var_bootstrap_results_b1000_cosweights_varimax.bin'
#FILE_NAME_COMPS_SURR = 'results/slp_all_var_surr_bootstrap_results_b1000_cosweights_varimax.bin'
#METHOD_NAME = 'URPCA'

#FILE_NAME_EIGS = 'results/sat_all_var_comp_count_cosweights.bin'
#FILE_NAME_COMPS = 'results/sat_all_var_bootstrap_results_b1000_cosweights_varimax.bin'
#FILE_NAME_COMPS_SURR = 'results/sat_all_var_surr_bootstrap_results_b1000_cosweights_varimax.bin'
#METHOD_NAME = 'URPCA'

FILE_NAME_EIGS = 'results/slp_sh_var_comp_count_cosweights.bin'
FILE_NAME_COMPS = 'results/slp_sh_var_bootstrap_results_b1000_cosweights_varimax.bin'
METHOD_NAME = 'URPCA'


def render_frequency_prevalence(gf, period, templ):
    """
    The frequency is in samples/year.
    """

    ff = np.zeros((gf.d.shape[1], gf.d.shape[2]), dtype = np.float64)
    tm = np.arange(0, gf.d.shape[0] / 12.0, 1.0 / 12, dtype = np.float64)
    for i in range(gf.d.shape[1]):
        for j in range(gf.d.shape[2]):
            pg = lombscargle(tm, gf.d[:, i, j].astype(np.float64), np.array([2.0 * np.pi / period]))
            ff[i,j] = np.sqrt(pg[0] * 4.0 / tm.shape[0])    

    f = render_component_single(ff, gf.lats, gf.lons, None, None, '%gyr period' % period)
    f.savefig('figs/%s_%dyr_cycle_prevalence.pdf' % (templ, period))

    

def render_component_oneframe(gf, templ):
    
    with open(FILE_NAME_COMPS, 'r') as f:
        d = cPickle.load(f)
        
    mn = d['mean']
    Nc = mn.shape[1]
    
    mn_mask = (np.abs(mn) > 1.0 / mn.shape[0]**0.5)
    mn_thr = mn * mn_mask
    zero_mask = np.all(mn_mask == False, axis = 1) 
    cid = np.argmax(np.abs(mn) * mn_mask, axis = 1)[:, np.newaxis] + 1
    cid[zero_mask, :] = 0
    
    mnd = gf.reshape_flat_field(cid)
    
    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(gf.lats)
    lats_s = gf.lats[lat_ndx]
    
    # shift the grid by 180 degs and remap lons to -180, 180
    Cout, lons_s = basemap.shiftgrid(180, mnd[0, :, :], gf.lons)
    lons_s -= 360
        
    fig = plt.figure()

    # construct the projection from the remapped data
    m = basemap.Basemap(projection='mill',
                llcrnrlat=lats_s[0], urcrnrlat=lats_s[-1],
                llcrnrlon=lons_s[0], urcrnrlon=lons_s[-1],
                resolution='c')
    
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,1])
    m.drawmeridians(np.arange(-120., 121.,60.), labels=[1,0,0,1])
   
    nx = int((m.xmax-m.xmin) / 20000) + 1
    ny = int((m.ymax-m.ymin) / 20000) + 1
    f = m.transform_scalar(Cout[lat_ndx, :], lons_s, lats_s, nx, ny, order = 0)
    
    plt.title('Components [%s]' % METHOD_NAME)
    imgplt = m.imshow(f, alpha = 0.8, cmap = plt.get_cmap('Paired'))
    
    fig.savefig('figs/%s_component_clusters.pdf' % templ, bbox_inches = 'tight', pad_inches = 0.5, transparent = True)
    
    print('Uncovered area: %d grid points' % (np.sum(cid==0)))
    
    
    c_size = np.zeros((Nc+1,))
    for i in range(Nc+1):
        c_size[i] = np.sum(cid == i)
        
    f = plt.figure()
    plt.title('Cluster sizes')
    plt.bar(np.arange(Nc) - 0.3 + 1, c_size[1:], 0.6)
    plt.xlabel('Cluster id')
    plt.ylabel('Grid points in cluster')
    f.savefig('figs/%s_cluster_sizes.pdf' % templ)
    
    
def render_slp_component_element_values():
    
    with open(FILE_NAME_COMPS, 'r') as f:
        d = cPickle.load(f)
        
    mn = d['mean']
    
    cid = np.amax(np.abs(mn), axis = 1)[:, np.newaxis]
    
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.slice_spatial(None, [20, 89])
    mnd = gf.reshape_flat_field(cid)
    
    f = render_component_single(mnd[0, :, :], gf.lats, gf.lons, (0, 0.15), None,
                                'NH Extratropical Components - max values')
    f.savefig('figs/slp_nh_component_maxima_sameaxis.pdf')

    f = render_component_single(mnd[0, :, :], gf.lats, gf.lons, None, None,
                                'NH Extratropical Components - max values')
    f.savefig('figs/slp_nh_component_maxima.pdf')
    
    f = plt.figure()
    plt.hist(cid, bins = 40)
    plt.title('Histogram of max values @ grid points across components')
    plt.xlabel('Maximum value [-]')
    plt.ylabel('Frequency [-]')
    f.savefig('figs/slp_nh_compmax_hist.pdf')
    
    return f


def plot_components_stability_b1000(templ):
    
    with open(FILE_NAME_COMPS, 'r') as f:
        d = cPickle.load(f)
        
    mn = d['mean']

    with open(FILE_NAME_COMPS_SURR, 'r') as f:
        d = cPickle.load(f)
        
    mns = d['mean']
    
    Cstab = np.sum(mn**2, axis = 0)
    Csstab = np.sum(mns**2, axis = 0)
    
    f = plt.figure()
    plt.plot(np.arange(mn.shape[1]) + 1, Cstab, 'ro-')
    plt.plot(np.arange(mns.shape[1]) + 1, Csstab, 'go-')
    plt.axis([0.5, mns.shape[1] + 0.5, 0.0, 1.0])
    plt.title('Squared 2-norm of vectors')
    plt.legend(('B1000', 'Surrogate'))
    f.savefig('figs/%s_component_stability.pdf' % templ)
    
    print('Mean component stability: %g (std %g)' % (np.mean(Cstab), np.std(Cstab)))
#    print('Mean surrogate stability: %g (std %g)' % (np.mean(Csstab), np.std(Csstab)))
    
    
def render_slp_model_orders():
    
    with open(FILE_NAME_EIGS, 'r') as f:
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

    f.savefig('figs/slp_nh_model_order.pdf')
    
    
def plot_slp_component_eigvals():
    
    with open(FILE_NAME_EIGS, 'r') as f:
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
    
    f = plt.figure()
    
    plt.title('SLP NH Extratropical - Eigvals from data and surrogates')    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')

    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    plt.xlabel('Eigenvalue index [-]')
    plt.ylabel('Eigenvalue magnitude [-]')
    
    plt.legend(('Data', 'AR', 'WN', 'Fourier'))
    
    plt.axes([0.3, 0.3, 0.5, 0.3], axisbg = 'w')
    plt.axis([55, 70, 10, 23])
    
    plt.plot(np.arange(N) + 1, dlam, 'ro-', linewidth = 1.5)
    mn = np.mean(slam_ar, axis = 0)
    plt.errorbar(np.arange(N) + 1, mn, np.vstack([mn - slam_ar[p0025, :], slam_ar[p0975, :] - mn]), fmt = 'bo-')

    mn = np.mean(slam_w1, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.1, mn, np.vstack([mn - slam_w1[p0025, :], slam_w1[p0975, :] - mn]), fmt = 'go-')
    
    mn = np.mean(slam_f, axis = 0)
    plt.errorbar(np.arange(N) + 1 + 0.2, mn, np.vstack([mn - slam_f[p0025, :], slam_f[p0975, :] - mn]), fmt = 'mo-')
    
    f.savefig('figs/slp_nh_eigenvalues.pdf')


def render_set_stub(x):
    render_component_set(*x)


def ls_spectral_estimate(freqs, tsn):
    """
    The frequencies must be in angular "months".
    """
    ls = lombscargle(np.arange(tsn.shape[0], dtype = np.float64), tsn, freqs)
    return np.sqrt(ls * 4.0 / tsn.shape[0])


def render_components(gf, templ):

    with open(FILE_NAME_COMPS, 'r') as f:
        d = cPickle.load(f)

    mn = d['mean']
    mn /= np.sum(mn**2, axis = 0)**0.5
    mn_gf = gf.reshape_flat_field(mn)
    
    data = gf.data()
    data = np.transpose(np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2])))
    
    # extract the time series
    ts = np.dot(mn.T, data).astype(np.float64)
    tsn = ts - np.mean(ts, axis = 1)[:, np.newaxis]
    tsn /= np.std(tsn, axis = 1)[:, np.newaxis]
    
    t_start = datetime.now()
    
    thr = 1.0 / mn.shape[0]**0.5
    
    periods = np.linspace(2.0, 240.0, 238*10)
    angular_freqs = 2.0 * np.pi * 1.0 / periods

    pool = Pool(4)
    render_list_sets = [ ([mn_gf[i, ...] * (np.abs(mn_gf[i,...]) > thr),
                           ('date', gf.tm, ts[i, :]),
                           ('invfreq', periods / 12.0, ls_spectral_estimate(angular_freqs, tsn[i, :])),
                           ('plot', np.arange(ts.shape[1]) - ts.shape[1] / 2,
                            1.0 / tsn.shape[1] * np.correlate(tsn[i,:], tsn[i,:], mode = 'same'))],
                          [ 'Mean:Thr', 'Time series', 'Frequency', 'Autocorrelation'],
                          gf.lats, gf.lons, 'symm',
                          'figs/%s_comp%02d_varimax_mn.png' % (templ, i+1),
                          'Component %d' % (i+1))
                          for i in range(mn.shape[1]) ]

    # use less nodes to render the maps due to memory constraints
    map(render_set_stub, render_list_sets)
    pool.close()

    # clean up memory
    del render_list_sets

    print("DONE after [%s]." % (datetime.now() - t_start))
    
    
def plot_explained_variance(gf, templ):

    with open(FILE_NAME_COMPS, 'r') as f:
        d = cPickle.load(f)

    mn = d['mean']
#    mn /= np.sum(mn**2, axis = 0)**0.5
    
    data = gf.data()
    data *= gf.qea_latitude_weights()
    data = np.transpose(np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2])))
    
    # extract the time series
    ts = np.dot(mn.T, data)
    
    vars = np.sum(ts**2, axis = 1)
    
    total_var = np.trace(np.dot(data.T, data))

    plt.figure()
    plt.plot(np.arange(vars.shape[0]) + 1, vars / total_var)
    plt.plot([1, vars.shape[0]], [1.0 / vars.shape[0]] * 2, 'r-')
    plt.title('Explained variance of each component')
    plt.xlabel('Component index')
    plt.ylabel('Explained variance [-]')
    plt.savefig('figs/%s_explained_variance.png' % templ)

    print('Sum of explained variance %g' % (np.sum(vars) / total_var))

def plot_nao_correlations():
    
    # load geo-field
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.transform_to_anomalies()
    gf.normalize_monthly_variance()
    gf.slice_spatial(None, [20, 89])
    gf.slice_date_range(date(1950, 1, 1), date(2012, 3, 1))
    
    with open(FILE_NAME_COMPS, 'r') as f:
        d = cPickle.load(f)
        
    # unroll the data
    data = gf.data()
    data = np.transpose(np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2])))
    
    # load the monthly NAO index
    nao = np.loadtxt('data/nao_index.tim.txt', skiprows = 0)

    naoh = np.loadtxt('data/nao_index_hurrel.tim.txt', skiprows = 0)
    naoh_ndx = naoh[:, 2]
    nao_ndx = nao[:, 2]

    print nao_ndx.shape
    print naoh_ndx.shape
    
    ts_len = min(len(nao_ndx), len(naoh_ndx))

    nao_ndx = nao_ndx[:ts_len]
    naoh_ndx = naoh_ndx[:ts_len]
    
    mn = d['mean']
    mn = mn / np.sum(mn**2, axis = 0) ** 0.5
    
    Nc = mn.shape[1]
    
    ts = np.transpose(np.dot(mn.T, data))
    ts = ts[:ts_len, :]
    
    Cnao = np.zeros((Nc,))
    Cnaoh = np.zeros((Nc,))
    
    for i in range(Nc):
        Cnao[i] = np.corrcoef(nao_ndx, ts[:, i], rowvar = False)[0,1]
        Cnaoh[i] = np.corrcoef(naoh_ndx, ts[:, i], rowvar = False)[0,1]
    
    f = plt.figure()
#    plt.plot(nao_ndx, 'r-')
#    plt.plot(naoh_ndx, 'b-')
    plt.plot(np.arange(Nc) + 1, Cnao, 'ro-')
    plt.plot(np.arange(Nc) + 1, Cnaoh, 'go-')
    plt.legend(('NAO/PC', 'NAO/Stat.'))
    plt.xlabel('Component index [-]')
    plt.ylabel('NAO correlation [-]')
    
    f.savefig('figs/slp_nh_nao_correlation.pdf')
    
    print('Max station NAO correlation: %g at %d' % (np.amax(np.abs(Cnaoh)), np.argmax(np.abs(Cnaoh))))
    print('Max PC/NAO correlation: %g at %d' % (np.amax(np.abs(Cnao)), np.argmax(np.abs(Cnao))))
    
    
def plot_all_stabilities():
    
    with open('results/slp_nh_var_bootstrap_results_b1000_cosweights_fasticaT.bin', 'r') as f:
        d = cPickle.load(f)
        
    mn_fasticaT = np.sum(d['mean'] ** 2, axis = 0) ** 0.5
    
    Nc = mn_fasticaT.shape[0]

    with open('results/slp_nh_var_surrogate_bootstrap_results_b1000_cosweights.bin', 'r') as f:
        d = cPickle.load(f)
        
    mn_surr_model = np.sum(d['mean'] ** 2, axis = 0) ** 0.5
    
    with open('results/slp_nh_var_bootstrap_results_b1000_cosweights_fastica.bin', 'r') as f:
        d = cPickle.load(f)

    mn_fasticaS = np.sum(d['mean'] ** 2, axis = 0) ** 0.5
    
    with open('results/slp_nh_var_bootstrap_results_b1000_cosweights_spca.bin', 'r') as f:
        d = cPickle.load(f)

    mn_spca = np.sum(d['mean'] ** 2, axis = 0) ** 0.5

    with open('results/slp_nh_var_bootstrap_results_b1000_cosweights.bin', 'r') as f:
        d = cPickle.load(f)

    mn_urpca = np.sum(d['mean'] ** 2, axis = 0) ** 0.5

    f = plt.figure()
    plt.plot(np.arange(Nc) + 1, mn_urpca, 'go-')
    plt.plot(np.arange(Nc) + 1, mn_fasticaS, 'bo-')
    plt.plot(np.arange(Nc) + 1, mn_fasticaT, 'mo-')
    plt.plot(np.arange(Nc) + 1, mn_spca, 'ko-')
    plt.plot(np.arange(Nc) + 1, mn_surr_model, 'ro-')
    plt.axis([0.5, Nc + 0.5, 0.0, 1.0])
    plt.title('2-norm of averaged components (stability)')
    leg = plt.legend(('URPCA', 'FastICA-S', 'FastICA-T', 'PCA-S', 'Surrogate model'), 'lower center', ncol = 3)
    plt.setp(leg.get_texts(), fontsize = 'small')
    f.savefig('figs/slp_nh_component_stability_all.pdf')
    
if __name__ == '__main__':
    
#    gf = geo_data_loader.load_monthly_sat_all2()
#    templ = 'sat_all'

#    gf = geo_data_loader.load_monthly_slp_all2()
#    templ = 'slp_all'

#    gf = geo_data_loader.load_monthly_slp_sh()
#    templ = 'slp_sh'

    gf = geo_data_loader.load_monthly_sat_sh()
    templ = 'sat_sh'
   

#    gf = geo_data_loader.load_daily_slp_sh()
#    templ = 'slp_sh'

#    plot_slp_component_eigvals()
#    render_slp_component_element_values()
    
#    plot_components_stability_b1000(templ)
#    render_component_oneframe(gf, templ)
#        
#    plot_nao_correlations()
#    plt.show()
    
    render_components(gf, templ)
    
#    plot_explained_variance(gf, templ)
    
#    plot_all_stabilities()

    for p in np.arange(1.0 + 0.01, 12.0, dtype = np.float64):
        render_frequency_prevalence(gf, p, templ)
    
