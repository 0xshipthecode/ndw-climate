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
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros((m,))
    
    for i in range(N):
        
        # unroll the matrix multiplication manually serial CPU-style (here instead of np.dot)
        for j in range(m):
            v[j] = w[j] + ints[i, j]
            for k in range(m):
                v[j] += A[j, k] * u[k]

        # shift everything by m elements
        for j in range(mp-m, m, -1):
            u[j+m] = u[j]                

        # first m elements are copied from v
        for j in range(m):
            u[j] = v[j]
            outts[i, j] = v[j]
