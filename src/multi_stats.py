
import numpy as np


def compute_eigvals_pvalues(dlam, slam):
    """
    Computes the p-value of each eigenvalue given the data eigenvalues and the surrogate eigenvalues 
    from slam.shape[0] surrogates.  It is required that dlam.shape[0] == slam.shape[1].
    """
    # compute each p-val
    Nsurr = slam.shape[0]
    p_vals = np.zeros_like(dlam, dtype = np.float64)
    
    for i in range(len(p_vals)):
        p_vals[i] = float(np.sum(dlam[i] <= slam[:,i])) / Nsurr
        
    return p_vals
        

def bonferroni_test(p_vals, sig_level, Nsurr, Nhyp = None):
    """
    Run a Bonferroni multiple testing procedure on p-values <p_vals> with significance
    level <sig_level> given that <Nsurr> surrogates were used to compute the p-values.
    Optionally the number of simultaneously tested hypotheses can be set with <Nhyp>,
    if left at None, len(p_vals) will be used.
    """
    # number of hypothesis may be set externally for robustness testing purposes
    # i.e. does the number of components depende on the number of hypotheses?
    if Nhyp is None:
        Nhyp = len(p_vals)

    # if the number of hypotheses is higher than the Nhyp (can happen for robustness testing)
    # remove the tail elements of p_vals until p_vals has the length Nhyp
    if Nhyp < len(p_vals):
        p_vals = p_vals[:Nhyp]

    bonf_level = sig_level / Nhyp
    
    if bonf_level < 1.0 / Nsurr:
        raise "Will not run Bonferroni, not enough surrogates available for the test!"
    
    return p_vals < bonf_level


def sidak_test(p_vals, sig_level, Nsurr, Nhyp = None):
    """
    Run a Bonferroni-Sidak multiple testing procedure on p-values <p_vals> with significance
    level <sig_level> given that <Nsurr> surrogates were used to compute the p-values.
    Optionally the number of simultaneously tested hypotheses can be set with <Nhyp>,
    if left at None, len(p_vals) will be used.
    """
    # number of hypothesis may be set externally for robustness testing purposes
    # i.e. does the number of components depende on the number of hypotheses?
    if Nhyp is None:
        Nhyp = len(p_vals)

    # if the number of hypotheses is higher than the Nhyp (can happen for robustness testing)
    # remove the tail elements of p_vals until p_vals has the length Nhyp
    if Nhyp < len(p_vals):
        p_vals = p_vals[:Nhyp]

    sidak_level = 1.0 - (1.0 - sig_level) ** (1.0 / Nhyp)
    
    if sidak_level < 1.0 / Nsurr:
        raise "Will not run Bonferroni-Sidak test, not enough surrogates available for the test!"
    
    return p_vals < sidak_level


def fdr_test(p_vals, sig_level, Nsurr, Nhyp = None):
    """
    Run an FDR multiple testing procedure on p-values <p_vals> with significance
    level <sig_level> given that <Nsurr> surrogate were used to compute the p-values.
    Optionally the number of simultaneously tested hypotheses can be set with <Nhyp>,
    if left at None, len(p_vals) will be used.

    NOTE: if Nhyp < len(p_vals), the p_vals tail will be chopped off so that len(p_val) = Nhyp.
          Then only will the test be run.
    """
    # number of hypothesis may be set externally for robustness testing purposes
    # i.e. does the number of components depende on the number of hypotheses?
    if Nhyp is None:
        Nhyp = len(p_vals)

    # if the number of hypotheses is higher than the Nhyp (can happen for robustness testing)
    # remove the tail elements of p_vals until p_vals has the length Nhyp
    if Nhyp < len(p_vals):
        p_vals = p_vals[:Nhyp]

    # the sorting is done only after the number of p_vals is fixed, see comments.
    sndx = np.argsort(p_vals)
    
    bonf_level = sig_level / Nhyp
    
    if bonf_level < 1.0 / Nsurr:
        raise "Will not run FDR, not enough surrogates used for the test!"
    
    Npvals = len(p_vals)
    h = np.zeros((Npvals,), dtype = np.bool)
    
    # test the p-values in order of p-values (smallest first)
    for i in range(Npvals - 1, 0, -1):
        
        # select the hypothesis with the i-th lowest p-value
        hndx = sndx[i]
        
        # check if we satisfy the FDR condition
        if p_vals[hndx] <= (i+1)*bonf_level:
	    h[sndx[:i+1]] = True
            break
        
    return h


def holm_test(p_vals, sig_level, Nsurr, Nhyp = None):
    """
    Run a Bonferroni-Holm multiple testing procedure on p-values <p_vals> with significance
    level <sig_level> given that <Nsurr> surrogate were used to compute the p-values.
    Optionally the number of simultaneously tested hypotheses can be set with <Nhyp>,
    if left at None, len(p_vals) will be used.

    NOTE: if Nhyp < len(p_vals), the p_vals tail will be chopped off so that len(p_val) = Nhyp.
          Then only will the test be run.
    """
    # number of hypothesis may be set externally for robustness testing purposes
    # i.e. does the number of components depende on the number of hypotheses?
    if Nhyp is None:
        Nhyp = len(p_vals)

    # if the number of hypotheses is higher than the Nhyp (can happen for robustness testing)
    # remove the tail elements of p_vals until p_vals has the length Nhyp
    if Nhyp < len(p_vals):
        p_vals = p_vals[:Nhyp]

    # the sorting is done only after the number of p_vals is fixed, see comments.
    sndx = np.argsort(p_vals)
    
    bonf_level = sig_level / Nhyp
    
    if bonf_level < 1.0 / Nsurr:
        raise "Will not run Bonferroni-Holm test, not enough surrogates used for the test!"

    h = np.zeros((Nhyp,), dtype = np.bool)

    # test the p-values in order of p-values (smallest first)
    for i in range(Nhyp):
        
        # select the hypothesis with the i-th lowest p-value
        hndx = sndx[i]
        
        # check if we have violated the Bonf-Holm condition
        if p_vals[hndx] > sig_level / (Nhyp - i):
            break
        
        # the hypothesis is true
        h[hndx] = True
            
    return h

