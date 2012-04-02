
import numpy as np


def estimate_snr(S, U):
    """
    Estimate compute the difference between means of signal and noise divided by the
    sum of standard deviations of the signal and noise for each component.  S is the
    ideal form of the components (as unit vectors).
    """
    N, M = U.shape
    C = S.shape[1]
        
    # compute the SNR for each component (which is now matched according to index i)
    snr = np.zeros((C,))
    for i in range(C):
        mean_sig = np.mean(U[S[:,i] > 0, i])
        std_sig = np.std(U[S[:,i] > 0, i])
        mean_noise = np.mean(U[S[:,i] == 0, i])
        std_noise = np.std(U[S[:,i] == 0, i])
        snr[i] = (mean_sig - mean_noise) / (std_sig + std_noise)
        
    return snr


def mse_error(S, U):
    """
    Compute the mean square error between the template and the
    computed components.  S is the ideal form of the components (unit vectors).
    """
    # compute the ssq for each component (which is now matched according to index i)
    mse = np.mean((U - S)**2, axis = 0)
        
    return mse


def marpe_error(S, U):
    """
    Compute the mean absolute relative prediction error between the template and the computed
    components.  S is the ideal form of the components (unit vectors).  Returns fractions.
    """
    mape = np.mean(np.abs(U - S), axis = 0)
    
    return mape / np.max(S, axis = 0)