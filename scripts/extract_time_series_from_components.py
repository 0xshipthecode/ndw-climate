

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
    
    sio.savemat('results/nao_correlation_components.mat', { 'Cnao_minus' : Cminus, 'Cnao_plus' : Cplus})
    
    plt.figure()
    plt.subplot(131)
    p = plt.imshow(Cminus, interpolation = 'nearest')
    p.set_clim(-0.3, 1.0)
    plt.colorbar()
    plt.title('NAO-')
    plt.subplot(132)
    p = plt.imshow(Cplus, interpolation = 'nearest')
    p.set_clim(-0.3, 1.0)
    plt.colorbar()
    plt.title('NAO+')
    plt.subplot(133)
    p = plt.imshow(Cplus - Cminus, interpolation = 'nearest')
    p.set_clim(-0.25, 0.25)
    plt.colorbar()
    plt.title('Correlation differences')
    plt.show()
     
    
    