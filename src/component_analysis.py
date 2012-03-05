

import numpy as np
from scipy.linalg import svdvals

def pca_eigvals(D):
    """Compute the PCA of a geo-field that will be unrolled into one dimension.
       axis[0] must be time, other axes are considered spatial and will be unrolled"""
       
    # reshape by combining all spatial dimensions
    d = D.copy()
    d = np.reshape(d, (d.shape[0], np.prod(d.shape[1:])))
    
    # we need the constructed single spatial dimension to be on axis 0
    d = d.transpose()
    
    # remove mean of each time series
    d = d - np.mean(d, axis = 1)[:, np.newaxis]
    
    return 1.0 / (d.shape[1] - 1) * svdvals(d, True)**2
