import numpy as np
cimport numpy as np
cimport cython

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.cdivision(True)
def execute_Aupw(np.ndarray[DTYPE_t, ndim=2] A not None,
                  np.ndarray[DTYPE_t, ndim=1] w not None,
                  np.ndarray[DTYPE_t, ndim=1] u not None,
                  np.ndarray[DTYPE_t, ndim=2] ints not None,
                  np.ndarray[DTYPE_t, ndim=2] outts not None):
    
    cdef int i, j, k
    cdef int N = ints.shape[0]
    cdef int m = A.shape[0]
    cdef int mp = A.shape[1]
    cdef int p = mp // m
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros((m,), dtype = DTYPE)
    
    for i in range(N):
        
        # v[i] = A * u[i]
        # unroll the matrix multiplication manually serial CPU-style (here instead of np.dot)
        for j in range(m):
            v[j] = w[j] + ints[i, j]
            for k in range(mp):
                v[j] += A[j, k] * u[k]

        # shift everything in predictor u by m elements "down" (back in time)
        for j from mp-m > j >= 0:
            u[j+m] = u[j]

        # first m elements of u are copied from v (prediction of model)
        # vis stored in output ts
        for j in range(m):
            u[j] = v[j]
            outts[i, j] = v[j]
