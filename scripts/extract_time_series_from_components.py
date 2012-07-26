
from datetime import date
from geo_field import GeoField
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits import basemap
import scipy.io as sio


FILE_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights.bin'
METH_NAME = 'URPCA'

#FILE_COMPS = 'results/slp_nh_var_bootstrap_results_b1000_cosweights_fastica.bin'
#METH_NAME = 'FastICA'


def get_superthresh_links(C, frac):
    """
    Obtain the set of superthreshold links in (from, to) tuple.  Index is one-based so
    that the indices are components directly.
    """
    N = C.shape[0]
    
    # extract upper triangle
    i1 = np.triu_indices(N, 1)
    vals = np.abs(C[i1])

    # find the threshold value    
    vals.sort()
    thr_ndx = int((1.0 - frac) * len(vals))
    thr = vals[thr_ndx]
    
    links = []
    for i in range(N):
        for j in range(i+1, N):
            if abs(C[i,j]) >= thr:
                col = 'r' if C[i,j] < 0 else 'k' 
                links.append((i+1, j+1, col))
    
    return links


def threshold_matrix(C, frac):
    
    N = C.shape[0]
    
    # extract upper triangle
    i1 = np.triu_indices(N, 1)
    vals = np.abs(C[i1])

    # find the threshold value    
    vals.sort()
    thr_ndx = int((1.0 - frac) * len(vals))
    thr = vals[thr_ndx]

    return C >= thr



def find_component_center(mask):
    
    b = mask.copy()
    
    nzy, nzx = np.nonzero(mask)
    nzy = nzy[::-1]
    nzx = nzx[::-1]
    b_set = len(nzx)
    while(b_set > 1):
        
        # remove by erosion all non-internal locations
        for y,x in zip(nzy, nzx):
            if np.sum(mask[y-1:y+2, x-1:x+2]) < 9 and b_set > 1:
                b[y, x] = 0
                b_set -= 1
                
        mask[:] = b[:]
        nzy, nzx = np.nonzero(mask)
        nzy = nzy[::-1]
        nzx = nzx[::-1]
        b_set = len(nzx)
        
    cy, cx = np.nonzero(b)
    return cx[0], cy[0]


def find_component_centers_on_grid(gf, clusts):
    
    # mark maxima
    mx_pos = np.argmax(mn**2, axis = 0)
    
#    mx_loc = np.zeros_like(mn)
#    mx_loc[(mx_pos, np.arange(mn.shape[1]))] = 1.0
#    
#    mx_loc_gf = gf.reshape_flat_field(mx_loc)
    
    lats = []
    lons = []
    
    for c in range(mn.shape[1]):
        x, y = find_component_center(clusts == c+1)
        lats.append(gf.lats[y])
        lons.append(gf.lons[x])
        
    lons = np.array(lons)
    lons[lons >= 180] = lons[lons >= 180] - 360
    return lons, np.array(lats)
    

def find_clusters_from_components(gf, mn):
    
    Nc = mn.shape[1]
    
    mn_mask = (np.abs(mn) > 1.0 / mn.shape[0]**0.5)
    mn_thr = mn * mn_mask
    zero_mask = np.all(mn_mask == False, axis = 1) 
    cid = np.argmax(np.abs(mn) * mn_mask, axis = 1)[:, np.newaxis] + 1
    cid[zero_mask, :] = 0
    
    mnd = gf.reshape_flat_field(cid)
    return mnd[0, :, :]


def plot_clusters_with_centers_and_links(gf, clusts, c_lons, c_lats, links = []):
    
    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(gf.lats)
    lats_s = gf.lats[lat_ndx]
    
    # shift the grid by 180 degs and remap lons to -180, 180
    Cout, lons_s = basemap.shiftgrid(180, clusts, gf.lons)
    lons_s -= 360
    
    plt.figure()
    plt.title('NH Extratropical Components [%s]' % METH_NAME)
        
    # construct the projection from the remapped data
    m = basemap.Basemap(projection='mill',
                llcrnrlat=lats_s[0], urcrnrlat=lats_s[-1],
                llcrnrlon=lons_s[0], urcrnrlon=lons_s[-1],
                resolution='c')
    
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,91.,30.), labels = [1,0,0,1])
    m.drawmeridians(np.arange(-120., 121.,60.), labels = [1,0,0,1])
   
    nx = int((m.xmax-m.xmin) / 20000) + 1
    ny = int((m.ymax-m.ymin) / 20000) + 1
    f = m.transform_scalar(Cout[lat_ndx, :], lons_s, lats_s, nx, ny, order = 0)
    
    imgplt = m.imshow(f, alpha = 0.9, cmap = plt.get_cmap('Paired'))
    
    # plot the centers of the components
    x, y = m(c_lons, c_lats) 
    plt.plot(x, y, 'ko', markersize = 6)
    
    # plot the links, if any
    for n1, n2, c in links:
        plt.plot([x[n1-1], x[n2-1]], [y[n1-1], y[n2-1]], '%s-' % c, alpha = 0.8)
        
    plt.savefig('figs/slp_nh_component_clusters_with_centers.pdf', bbox_inches = 'tight', pad_inches = 0.5)
        

if __name__ == '__main__':
    
    # load geo-field
    gf = GeoField()
    gf.load('data/pres.mon.mean.nc', 'pres')
    gf.transform_to_anomalies()
    gf.normalize_monthly_variance()
    gf.slice_spatial(None, [20, 89])
    gf.slice_date_range(date(1950, 1, 1), date(2012, 3, 1))
    
    # unroll the data
    data = gf.data()
    data = np.transpose(np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2])))
    
    # load the monthly NAO index
    nao = np.loadtxt('data/nao_index.tim.txt', skiprows = 0)
    print(nao.shape)
    nao_ndx = nao[:745, 2]
    
    naoh = np.loadtxt('data/nao_index_hurrel.tim.txt', skiprows = 0)
    print(naoh.shape)
    naoh_ndx = naoh[:745, 2]

    # load the components
    with open(FILE_COMPS, 'r') as f:
        d = cPickle.load(f)
        
    # convert to unit vectors
    mn = d['mean']
    
    c_field = find_clusters_from_components(gf, mn)
    
    # find max values of components and plot them on top of a graph
    cc_lons, cc_lats = find_component_centers_on_grid(gf, c_field)
    
    mn = mn / np.sum(mn**2, axis = 0) ** 0.5
    
    ts = np.transpose(np.dot(mn.T, data))
    print(ts.shape)
    
    sio.savemat('results/slp_nh_component_time_series.mat', { 'ts' : ts})
    
    nao_pos = np.argsort(nao_ndx)
    
    f30 = 745 // 3
    
    data_naominus = ts[nao_pos[:f30], :]
    data_naoplus = ts[nao_pos[-f30:], :]
    
    print(data_naominus.shape)
    print(data_naoplus.shape)
    
    Cminus = np.corrcoef(data_naominus, rowvar = False)
    Cplus = np.corrcoef(data_naoplus, rowvar = False)
    
    Dminus = threshold_matrix(Cminus, 0.1)
    Dplus = threshold_matrix(Cplus, 0.1)
    
    sio.savemat('results/nao_correlation_components_01.mat', { 'Cnao_minus' : Cminus, 'Cnao_plus' : Cplus,
                                                            'Dnao_minus' : Dminus, 'Dnao_plus' : Dplus})
    diag1 = np.eye(Cminus.shape[0])
    
#    plt.figure()
#    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
#    plt.subplot(131)
#    p = plt.imshow(Cminus**2 - diag1, interpolation = 'nearest')
#    p.set_clim(0.0, 0.5)
#    plt.colorbar(fraction = 0.07, shrink = 0.7, aspect = 15)
#    plt.title('NAO- squared corr. [zero diag]', fontsize = 18)
#    plt.subplot(132)
#    p = plt.imshow(Cplus**2 - diag1, interpolation = 'nearest')
#    p.set_clim(0.0, 0.5)
#    plt.colorbar(fraction = 0.07, shrink = 0.7, aspect = 15)
#    plt.title('NAO+ squared corr. [zero diag]', fontsize = 18)
#    plt.subplot(133)
#    p = plt.imshow((Cplus - Cminus)**2, interpolation = 'nearest')
#    p.set_clim(-0.14, 0.14)
#    plt.colorbar(fraction = 0.07, shrink = 0.7, aspect = 15)
#    plt.title('Squared differences of correlations', fontsize = 18)
#    plt.show()
    
    print(nao_ndx.shape)
    print(ts.shape)
    
    Cp = np.corrcoef(np.hstack([nao_ndx[:732, np.newaxis], ts[:732,:]]), rowvar = False)
    Cp = Cp[:, 0]
    Ch = np.corrcoef(np.hstack([naoh_ndx[:732, np.newaxis], ts[:732,:]]), rowvar = False)
    Ch = Ch[:, 0]
    plt.figure()
    plt.plot(np.arange(ts.shape[1]) + 1, Cp[1:])
    plt.plot(np.arange(ts.shape[1]) + 1, Ch[1:])
    plt.legend(('Corr with PC NAO', 'Corr with Station NAO'))
    plt.title('Correlation of component time series with NAO index')
    plt.savefig('figs/nao_correlations.png')
    
    print np.argsort(Ch)[-5:]+1
    
    Cdiff = (Cplus - Cminus)

    plot_clusters_with_centers_and_links(gf, c_field, cc_lons, cc_lats, [])
    
    md = sio.loadmat('results/nao_modularity_01.mat')
    Cm = md['Cm']
    Cp = md['Cp'] 
    c_fieldp = np.zeros_like(c_field)
    c_fieldm = np.zeros_like(c_field)

    for i in range(1, 46):
        c_fieldp[c_field == i] = Cp[i-1]
        c_fieldm[c_field == i] = Cm[i-1]
    
    plot_clusters_with_centers_and_links(gf, c_fieldp, [], [], [])    
    plot_clusters_with_centers_and_links(gf, c_fieldm, [], [], [])    
#    plot_clusters_with_centers_and_links(gf, c_field, cc_lons, cc_lats, get_superthresh_links(Cplus, 0.1))
#    plot_clusters_with_centers_and_links(gf, c_field, cc_lons, cc_lats, get_superthresh_links(Cminus, 0.1))
#    plot_clusters_with_centers_and_links(gf, c_field, cc_lons, cc_lats, get_superthresh_links(Cdiff, 0.05))

    plt.show()
