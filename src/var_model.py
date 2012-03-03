# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:59:39 2012

@author: martin

Programmed according to Neumaier and Schneider and heavily inspired by ARFIT toolkit.
"""

import numpy as np
from scipy import linalg
import math
import var_model_acc


class VARModel:
    """A VAR(k) model with coefficient matrices.  The model is specified by
       a mean value (w), a coefficient matrix (A) and by a Cholesky factor
       of the noise covariance (L).
       """
    
    
    def __init__(self):
        """Initialize as empty model."""
        self.A = None
        self.w = None
        self.Lt = None
        
        
    def order(self):
        """Return model order."""
        return len(self.A) if self.A != None else 0
    
    
    def simulate_with_residuals(self, residuals, ndisc = 100):
        """Simulate a time series using this model using the supplied residuals
           in residuals (ndarray).  This function is NOT deterministic and employs
           a spin-up phase to init the model state using random noise."""
           
        A = self.A
        w = self.w
        
        m, p = A.shape[0], A.shape[1] / A.shape[0]
        N = residuals.shape[0]
        ts = np.zeros((N, m))

        u = np.zeros((m*p,))
        
        # initialize system using noise with correct covariance matrix 
        eps_noise = np.dot(np.random.normal(size=(ndisc, m)), self.Lt)
        
        # spin-up to random state
        # this is a HACK, the input and output arrays are the same. this exploits the fact
        # that (1) eps_noise is not used anymore and (2) the overwriting does not affect the computation
        # see function execute_Aupw.  The vector u ends up being the last state of the VAR model 
        var_model_acc.execute_Aupw(A, w, u, eps_noise, eps_noise)
        
#        for i in range(ndisc):
#            np.dot(A, u, out = v)
#            v += w
#            v += eps_noise[i, :]
#            u[m:] = u[:-m]
#            u[:m] = v
        
        # start feeding in residuals and store result in ts
        var_model_acc.execute_Aupw(A, w, u, residuals, ts)

#        for i in range(N):
#            v = np.dot(A, u, out = v)
#            v += residuals[i, :]
#            u[m:] = u[:-m]
#            u[:m] = v
#            ts[i, :] = v
            
        return ts
    

    def simulate(self, N, ndisc = 100):
        """Simulate process described by the model.  Obtain N samples and spin up the model for 100 steps."""
        
        A = self.A
        w = self.w
        
        m, p = A.shape[0], A.shape[1] / A.shape[0]
        ts = np.zeros((N, m))
        
        u = np.zeros((m*p,))
        v = np.zeros((m,))
        
        # construct noise with covariance matrix L^T * L
        eps_noise = np.dot(np.random.normal(size=(N + ndisc, m)), self.L.T)
        
        for i in range(ndisc):
            np.dot(A, u, out = v)
            v += w
            v += eps_noise[i, :]
            u[m:] = u[:-m]
            u[:m] = v
            
        for i in range(N):
            v = np.dot(A, u, out = v)
            v += eps_noise[i + ndisc, :]
            u[m:] = u[:-m]
            u[:m] = v
            ts[i, :] = v
            
        return ts


    def estimate(self, time_series, prng, fit_intercept = False, crit_type = 'sbc', ndx = None):
        """Stepwise selection of the model using QR decomposition according to Neumaier and Schneider across
           model orders [prng[0], prng[1]].  The criterion 'crit' is the model order selection criterion.
           Optionally the (integer) index array indicates which parts of the time series are contiguous and which are not."""
        
        p_min, p_max = prng[0], prng[1]
        fi = 1 if fit_intercept else 0
        ts = time_series[:, np.newaxis] if time_series.ndim == 1 else time_series
        N, m = ts.shape
        n_p = np.zeros(shape=(p_max+1,), dtype = np.int)
        n_p[p_max] = fi + p_max * m
        
        # remove "presample" data (p_max values from start) 
        N = N - p_max
        
        # construct matrix K (add row space for regularization matrix deltaD)
        K = np.zeros((N + n_p[p_max] + m, n_p[p_max] + m))
        
        # add intercept if required
        if(fit_intercept):
            K[:N, 0] = 1.0
            
        # set predictors u
        for j in range(1, p_max + 1):
            K[:N, fi+(j-1)*m:fi+j*m] = ts[p_max-j:N+p_max-j, :]
                
        # set predictors v
        K[:N, n_p[p_max]:n_p[p_max]+m] = ts[p_max:N+p_max, :]
        
        # add regularization as per paper of Neumaier & Schneider, who refer to Higham
        q = n_p[p_max] + m
        delta = (q**2 + q + 1) * np.finfo(np.float64).eps
        sc = (delta * np.sum(K**2, axis = 0))**0.5
        K[N:, :] = np.diag(sc) 

        # compute QR decomposition but only return R, Q is unused here 
        R = linalg.qr(K, True, -1, 'r')
        
        # retrieve R22 submatrix
        R22 = R[n_p[p_max]:q, n_p[p_max]:q] 
        
        # invert R22 matrix for later downdating
        invR22 = linalg.inv(R22)
        Mp = np.dot(invR22, invR22.T)
        
        # compute the log of the determinant of the residual cross product matrix
        logdp = np.zeros(p_max+1)
        sbc = np.zeros_like(logdp)
        fpe = np.zeros_like(logdp)
        logdp[p_max] = 2.0 * np.log(np.abs(np.prod(np.diag(R22))))
        
        # run the downdating steps & update estimates of log det covar mat
        q_max = q
        for p in range(p_max, p_min-1, -1):
            n_p[p] = m * p + fi
            q = n_p[p] + m
            
            # execute downdating step if at lower order
            if p < p_max:
                Rp = R[n_p[p]:q, n_p[p_max]:q_max]
                L = linalg.cholesky(np.identity(m) + np.dot(np.dot(Rp, Mp), Rp.T)).T
                Np = linalg.solve(L, np.dot(Rp, Mp))
                Mp -= np.dot(Np.T, Np)
                logdp[p] = logdp[p+1] + 2.0 * math.log(abs(np.prod(np.diag(L)))) 
        
            # compute selected criterion
            sbc[p] = logdp[p] / m - math.log(N) * (N - n_p[p]) / N
            fpe[p] = logdp[p] / m - math.log(N * float(N - n_p[p]) / float(N + n_p[p])) 

        # find the best order
        if crit_type == 'sbc':
            # find the best order
            p_opt = np.argmin(sbc[p_min:p_max+1]) + p_min
        elif crit_type == 'fpe':
            p_opt = np.argmin(fpe[p_min:p_max+1]) + p_min
        else:
            raise "Invalid criterion."
        
        # retrieve submatrices and intercept (if required)
        R11 = R[:n_p[p_opt], :n_p[p_opt]] 
        R12 = R[:n_p[p_opt], n_p[p_max]:n_p[p_max]+m]
        R22 = R[n_p[p_opt]:n_p[p_max]+m, n_p[p_max]:n_p[p_max]+m]    
        
        if n_p[p_opt] > 0:
            
            # improve conditioning
            if fit_intercept:
                scaler = np.max(sc[1:]) / sc[0]
                R11[:, 0] *= scaler
                
            Aaug = linalg.solve(R11, R12).T
            
            if fit_intercept:
                self.w = Aaug[:,0] * scaler
                self.A = Aaug[:, 1:n_p[p_opt]]
            else:
                self.w = np.zeros(shape=(m,))
                self.A = Aaug
                
        # compute estimate of covariance matrix and return it
        dof = N - n_p[p_opt]
        C = np.dot(R22.T, R22) / dof
        
        # store the Cholesky factor in L
        self.Lt = linalg.cholesky(C).transpose()
        
        return sbc, fpe
    
    
    def compute_residuals(self, time_series):
        """Retrieve the prediction residuals from the AR model of the time series."""
        w = self.w
        A = self.A
        
        ts = time_series[:, np.newaxis] if time_series.ndim == 1 else time_series
        m, p = A.shape[0], A.shape[1] / A.shape[0]
        N = ts.shape[0]
        Nres = N - p
        res = np.zeros((Nres, m))
        u = np.zeros((p*m,))
        
        # pre-fill predictors for first prediction
        for i in range(1, p+1):
            u[(i-1)*m:i*m] = ts[p-i, :]
            
        # copy time series ("residuals" without prediction)
        # subtract mean from all residuals
        res[:,:] = ts[p:N, :]
        res -= w
        
        Au = np.zeros_like(w)
        for ndx in range(p, N):
            # predict and subtract
            np.dot(A, u, out = Au)
            res[ndx-p] -= Au
            u[m:] = u[:-m]
            u[:m] = ts[ndx, :]
            
        # what remains are the residuals
        return res
        
    
    def pickle_model(self, p):
        """Save the model to file using the pickle libs."""
        p.dump(self, [self.w, self.A])
        

    def unpickle_model(self, up):
        """Load the model from the pickle file."""
        lst = up.load()
        self.w = lst[0]
        self.A = lst[1]
