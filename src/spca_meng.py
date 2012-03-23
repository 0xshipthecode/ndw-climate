

import numpy as np

"""
This module implementes the method of Meng et al 2012, Pattern recognition.
"""


def extract_sparse_component(X, k, init = None):
    """
    Extract a single sparse component from the data.  Optionally may be initialized
    by the parameter init.
    """
    N,M = X.shape
    
    # iniitialize with random vector
    if init is not None:
        w = init.copy()
    else:
        w = np.random.normal(size = (N,))
        
    w = w / np.sum(w**2) ** 0.5
    v = np.zeros_like(w)
    
    converged = False
    while not converged:

        v[:] = 0.0        
        for i in range(M):
            v += np.sign(np.dot(w, X[:,i])) * X[:,i]

        # find k+1 st top abs value and set everything up to it (including it)
        # to zero
        va = np.argsort(np.abs(v))
        v[va[:-k]] = 0.0
        
        # copy the k largest values to w and normalize w
        v /= np.sum(v**2) ** 0.5
        
        # check for convergence
        if np.all(v == w):
            converged = True
        else:
            for i in range(M):
                ortho_cond = np.dot(v, X[:,i]) == 0
                simult_nz_cond = np.dot(np.abs(np.sign(v)), np.abs(np.sign(X[:,i]))) != 0  
                if ortho_cond and simult_nz_cond:
                    v += np.random.normal(size = v.shape) * 0.1
                    v /= np.sum(v**2) ** 0.5
                    break
                
            # if we got here, we have converged
            converged = True
    
        w[:] = v[:]

    return w



def extract_sparse_components(X, k, m, init = None):
    """
    Extracts m components with sparsity k from the data X.  Optionally accepts
    an initial solution as init.  Variables are on axis [0], Observations on axis[1].
    """
    N, M = X.shape
    
    # copy & demean X
    X = X.copy()
    X = X - np.mean(X, axis = 1)[:, np.newaxis]
    
    # space for the results
    C = np.zeros((N, m))
    
    for i in range(m):
        c_i = extract_sparse_component(X, k, init[:,i] if init is not None else None)
        if np.sum(c_i != 0) > k:
            raise ValueError("Invalid component!")
        C[:,i] = c_i
        for j in range(M):
            X[:,j] -= np.dot(X[:,j], c_i) * c_i

    return C
    
