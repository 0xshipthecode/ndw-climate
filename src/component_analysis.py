

import numpy as np
from scipy.linalg import svdvals, svd


def pca_eigvals(d):
    """
    Compute the PCA of a geo-field that will be unrolled into one dimension.
    axis[0] must be time, other axes are considered spatial and will be unrolled
    so that the PCA is performed on a 2D matrix.
    """
    # reshape by combining all spatial dimensions
    d = np.reshape(d, (d.shape[0], np.prod(d.shape[1:])))
    
    # we need the constructed single spatial dimension to be on axis 0
    d = d.transpose()
    
    # remove mean of each time series
    d = d - np.mean(d, axis = 1)[:, np.newaxis]
    
    return 1.0 / (d.shape[1] - 1) * svdvals(d, True)**2



def pca_components(d):
    """
    Estimate the PCA components of a geo-field. d[0] must be time, other axes are considered spatial
    and will be unrolled so that the PCA is performed on a 2D matrix.
    """
    # reshape by combining all spatial dimensions
    d = np.reshape(d, (d.shape[0], np.prod(d.shape[1:])))
    
    # we need the constructed single spatial dimension to be on axis 0
    d = d.transpose()
    
    # remove mean of each time series
    d = d - np.mean(d, axis = 1)[:, np.newaxis]

    # svd will return only the diagonal of the S matrix    
    U, s, Vt = svd(d, False, True, True)
    s **= 2
    s *= 1.0 / (d.shape[1] - 1)
    
    # flip signs so that max(abs()) of each col is positive
    for i in range(U.shape[1]):
        if max(U[:,i]) < abs(min(U[:,i])):
            U[:,i] *= -1.0
            
    return U, s, Vt


def varimax(U, rtol = np.finfo(np.float32).eps, maxiter = 1000):
    """
    Rotate the matrix U using a varimax scheme.  Maximum no of rotation is 1000 by default.
    The rotation is in place (the matrix U is overwritten with the result).  For optimal performance,
    the matrix U should be contiguous in memory.  The implementation is based on MATLAB docs & code,
    algortithm is due to DN Lawley and AE Maxwell.
    """
    n,m = U.shape
    Ur = U.copy(order = 'C')
    ColNorms = np.zeros((1, m))
    
    dsum = 0.0
    for indx in range(maxiter):
        old_dsum = dsum
        np.sum(Ur**2, axis = 0, out = ColNorms[0,:])
        C = n * Ur**3
        C -= Ur * ColNorms  # numpy will broadcast on rows
        L, d, Mt = svd(np.dot(Ur.T, C), False, True, True)
        R = np.dot(L, Mt)
        dsum = np.sum(d)
        np.dot(U, R, out = Ur)
        if abs(dsum - old_dsum) / dsum < rtol:
            break
        
    # flip signs of components, where max-abs in col is negative
    for i in range(m):
        if np.amax(Ur[:,i]) < abs(np.amin(Ur[:,i])):
            Ur[:,i] *= -1.0

    return Ur, R, indx
