
import numpy as np
from var_model import VARModel
from geo_field import GeoField
import scipy.linalg


def constructVAR(S, cs, ar_rng, nghb_rng):
    """
    Based on a grid indicating which grid points are associated together
    in a cluster, the SMG constructs a VAR model that represents the spatial
    dependencies in the data runs the VAR model to generate model time series
    
    Construct an SMG based on the spatial matrix S and the cluster strengths cs.
    cs indicates for each cluster 1, 2, ... num_clusters what are the cross ar
    coefficients.  Each time series has autoregressive coefficient ar.
    
    Elements in S are processed row-wise.  The ravel()ed structural matrix is returned.
    """
    C = S.shape[1]
    Sr = S.ravel()  # automatically in C order (row-wise)
    N = Sr.shape[0]
    A = np.zeros(shape = (N,N), dtype = np.float64)
    w = np.zeros(shape = (N,), dtype = np.float64)
    
    # read the elements in C order (row by row)
    for i in range(N):
        A[i,i] = np.random.uniform(ar_rng[0], ar_rng[1])
        
        if Sr[i] > 0:
            blk_driver = np.nonzero(Sr == Sr[i])[0][0]
            if i > blk_driver:
                A[i, blk_driver] = cs[Sr[i]]
#                A[i, i] -= cs[Sr[i]]

    set_neighbor_weights(A, C, nghb_rng)
                
    # check stability of process
    if np.any(np.abs(scipy.linalg.eig(A, right = False)) > 1.0):
        raise ValueError("Unstable system constructed!")
     
    U = np.identity(N, dtype = np.float64)
    
    var = VARModel()
    var.set_model(A, w, U)
    return var, Sr


def set_neighbor_weights(A, C, cc):
    N = A.shape[0]
    # first set all neighbor correlations
    for i in range(N):
        ir, ic = i // C, i % C
        for j in range(N):
            jr, jc = j // C, j % C
            if abs(ir - jr) + abs(ic - jc) == 1:
                A[i,j] = np.random.uniform(cc[0], cc[1])
                

def get_neighbor_mask(N, C):
    M = np.zeros(shape=(N,N), dtype = np.bool)
    for i in range(N):
        ir, ic = i // C, i % C
        for j in range(N):
            jr, jc = j // C, j % C
            if abs(ir - jr) + abs(ic - jc) == 1:
                M[i,j] = True
    return M


def make_model_geofield(S, ts):
    
    M,N = S.shape
    T = ts.shape[0]

    gf = GeoField()
    gf.lats = np.arange(M)
    gf.lons = np.arange(N)
    gf.tm = np.arange(T)
    gf.d = np.reshape(ts, [T, M, N])

    return gf
