

import numpy as np
from scipy.linalg import svdvals, svd
from munkres import Munkres


def pca_eigvals_gf(d):
    """
    Compute the PCA of a geo-field that will be unrolled into one dimension.
    axis[0] must be time, other axes are considered spatial and will be unrolled
    so that the PCA is performed on a 2D matrix.
    """
    # reshape by combining all spatial dimensions
    d = np.reshape(d, (d.shape[0], np.prod(d.shape[1:])))
    
    # we need the constructed single spatial dimension to be on axis 0
    d = d.transpose()

    return pca_eigvals(d)


def pca_eigvals(d):
    """
    Compute the eigenvalues of the covariance matrix of the data d.  The covariance
    matrix is computed as d*d^T.
    """
    # remove mean of each row
    d = d - np.mean(d, axis = 1)[:, np.newaxis]
    
    return 1.0 / (d.shape[1] - 1) * svdvals(d, True)**2


def pca_components_gf(d, spatial = True):
    """
    Estimate the PCA components of a geo-field. d[0] must be time (observations).
    Other axes are considered spatial and will be unrolled into one variable dimension.
    The PCA is then performed on a 2D matrix with space (axis 1) as the variables and
    time (axis 0) as the observations.
    """
    # reshape by combining all spatial dimensions
    d = np.reshape(d, (d.shape[0], np.prod(d.shape[1:])))
    
    # we need the constructed single spatial dimension to be on axis 0
    d = d.transpose()
    
    return pca_components(d, spatial)


def pca_components(d, spatial = True):
    """
    Compute the standard PCA components assuming that axis0 are the variables (rows)
    and axis 1 are the observations (columns).  The data is not copied and is
    overwritten.
    """
    # remove mean of each time series
    if spatial:
        d = d - np.mean(d, axis = 1)[:, np.newaxis]
    else:
        d = d - np.mean(d, axis = 0)[np.newaxis, :]

    # SVD will return only the diagonal of the S matrix    
    U, s, Vt = svd(d, False, True, True)
    s **= 2
    s *= 1.0 / (d.shape[1] - 1)
    
    # flip signs so that max(abs()) of each col is positive
    for i in range(U.shape[1]):
        if max(U[:,i]) < abs(min(U[:,i])):
            U[:,i] *= -1.0
            Vt[i,:] *= -1.0
            
    return U, s, Vt


def corrmat_components(d):
    """
    Compute PCA components from the correlation matrix assuming that axis0 are the variables (rows)
    and axis 1 are the observations (columns).  The data is not copied and is overwritten.
    """
    # remove mean of each time series
    d = d - np.mean(d, axis = 1)[:, np.newaxis]

    # normalize each time series
    d = d / np.std(d, axis = 1)[:, np.newaxis]
    
    # svd will return only the diagonal of the S matrix    
    U, s, Vt = svd(d, False, True, True)
    s **= 2
    s *= 1.0 / (d.shape[1] - 1)
    
    # flip signs so that max(abs()) of each col is positive
    for i in range(U.shape[1]):
        if max(U[:,i]) < abs(min(U[:,i])):
            U[:,i] *= -1.0
            Vt[i,:] *= -1.0
            
    return U, s, Vt


def corrmat_components_gf(d):
    """
    Estimate the correlation matrix components of a geo-field. d[0] must be time (observations).
    Other axes are considered spatial and will be unrolled into one variable dimension.
    The PCA is then performed on a 2D matrix with space (axis 1) as the variables and
    time (axis 0) as the observations.
    """
    # reshape by combining all spatial dimensions
    d = np.reshape(d, (d.shape[0], np.prod(d.shape[1:])))
    
    # we need the constructed single spatial dimension to be on axis 0
    d = d.transpose()
    
    return corrmat_components(d)


def orthomax(U, rtol = np.finfo(np.float32).eps ** 0.5, gamma = 1.0, maxiter = 1000):
    """
    Rotate the matrix U using a varimax scheme.  Maximum no of rotation is 1000 by default.
    The rotation is in place (the matrix U is overwritten with the result).  For optimal performance,
    the matrix U should be contiguous in memory.  The implementation is based on MATLAB docs & code,
    algorithm is due to DN Lawley and AE Maxwell.
    """
    n,m = U.shape
    Ur = U.copy(order = 'C')
    ColNorms = np.zeros((1, m))
    
    dsum = 0.0
    for indx in range(maxiter):
        old_dsum = dsum
        np.sum(Ur**2, axis = 0, out = ColNorms[0,:])
        C = n * Ur**3
        if gamma > 0.0:
            C -= gamma * Ur * ColNorms  # numpy will broadcast on rows
        L, d, Mt = svd(np.dot(Ur.T, C), False, True, True)
        R = np.dot(L, Mt)
        dsum = np.sum(d)
        np.dot(U, R, out = Ur)
        if abs(dsum - old_dsum) / dsum < rtol:
            break
        
    # flip signs of components, where max-abs in col is negative
    for i in range(m):
        if np.amax(Ur[:,i]) < -np.amin(Ur[:,i]):
            Ur[:,i] *= -1.0
            R[i,:] *= -1.0
            
    return Ur, R, indx


def match_components_munkres(U1, U2):
    """
    Match the components from U1 to the components in U2 using the
    Hungarian algorithm.  The permutation which will bring U2 to match U1 is returned
    as the first element in the tuple.  Then U1 === U2[:, perm].
    The function also returns a sign_flip vector which can be applied to U2
    to switch the signs of the components to match those in U1.  The sign_flip,
    if applied, must be applied to U2 after the permutation!
    
    synopsis: perm, sf = match_components_munkres(U1, U2)
        
    """
    NC = U2.shape[1]
    
    # compute closeness of components using the dot product
    C = np.dot(U1.T, U2)

    # normalize dot product matrix by sizes (to compare unit size vectors)    
    U1s = 1.0 / np.sum(U1**2, axis = 0) ** 0.5
    U2s = 1.0 / np.sum(U2**2, axis = 0) ** 0.5
    C = U1s[:, np.newaxis] * C * U2s[np.newaxis, :]

    return match_components_from_matrix(C)
    

def match_components_from_matrix(C):
    """
    Find a maximal pairwise matching (using absolute value of C[i,j])
    between components, where the similarity between components is given
    by C.  C is typically either the correlation or dot product.
    """
    NC = C.shape[1]

    # find optimal matching of components
    m = Munkres()
    match = m.compute(1.0 - np.abs(C))
    perm = -1 * np.ones((NC,), dtype = np.int)
    sign_flip = np.zeros((1, NC), dtype = np.int)
    for i in range(len(match)):
        m_i = match[i]
        perm[m_i[0]] = m_i[1]
        sign_flip[0, m_i[0]] = -1 if C[m_i[0], m_i[1]] < 0.0 else 1.0
    
    return perm, sign_flip 


def matched_components(U1, U2):
    """
    Use the component matching method and return the matched components
    directly.  Matching is done to optimize order and polarity of component.
    The method also ensures the components have unit sizes.
    """
    C = U1.shape[1]
    
    # copy U, we will have to manipulate it
    U2 = U2.copy()

    # find the matching (even if there are more components in U)
    perm, sf = match_components_munkres(U1, U2)
    U2 = U2[:, perm[:C]]
    U2 *= sf[:, :C]
        
    U2 /= np.sum(U2**2, axis = 0) ** 0.5
    return U2
