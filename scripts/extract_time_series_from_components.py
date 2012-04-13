

from datetime import date
from geo_field import GeoField
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import scipy.io as sio

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
    
    # load the components
    with open('results/slp_nh_var_bootstrap_results_b1000.bin', 'r') as f:
        d = cPickle.load(f)
        
    # convert to unit vectors
    mn = d['mean']
    mn = mn / np.sum(mn**2, axis = 0) ** 0.5
    
    ts = np.transpose(np.dot(mn.T, data))
    print(ts.shape)
    
    nao_pos = np.argsort(nao_ndx)
    
    f30 = 745 // 3
    
    data_naominus = ts[nao_pos[:f30], :]
    data_naoplus = ts[nao_pos[-f30:], :]
    
    print(data_naominus.shape)
    print(data_naoplus.shape)
    
    Cminus = np.corrcoef(data_naominus, rowvar = False)
    Cplus = np.corrcoef(data_naoplus, rowvar = False)
    
#    sio.savemat('results/nao_correlation_components.mat', { 'Cnao_minus' : Cminus, 'Cnao_plus' : Cplus})
    
    plt.figure()
    plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95)
    plt.subplot(131)
    p = plt.imshow(Cminus**2 - np.eye(43), interpolation = 'nearest')
    p.set_clim(0.0, 0.5)
    plt.colorbar(fraction = 0.07, shrink = 0.7, aspect = 15)
    plt.title('NAO- squared corr. [zero diag]', fontsize = 18)
    plt.subplot(132)
    p = plt.imshow(Cplus**2 - np.eye(43), interpolation = 'nearest')
    p.set_clim(0.0, 0.5)
    plt.colorbar(fraction = 0.07, shrink = 0.7, aspect = 15)
    plt.title('NAO+ squared corr. [zero diag]', fontsize = 18)
    plt.subplot(133)
    p = plt.imshow((Cplus - Cminus)**2, interpolation = 'nearest')
    p.set_clim(-0.14, 0.14)
    plt.colorbar(fraction = 0.07, shrink = 0.7, aspect = 15)
    plt.title('Squared differences of correlations', fontsize = 18)
#    plt.show()
    
    print(nao_ndx.shape)
    print(ts.shape)
    
    C = np.corrcoef(np.hstack([nao_ndx[:, np.newaxis], ts]), rowvar = False)
    C = C[:, 0]
    plt.figure()
    plt.plot(np.arange(ts.shape[1]) + 1, C[1:])
    plt.title('Correlation of component time series with NAO index')
    plt.show()