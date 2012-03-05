# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:02:39 2012

@author: martin
"""


from geo_field import GeoField
from var_model import VARModel

import cPickle
import numpy as np


def _prepare_surrogates(a):
    i, j, ts = a
    v = VARModel()
    v.estimate(ts, [1, 30], True, 'sbc', None)
    r = v.compute_residuals(ts)
    return(i, j, v, r)


def _generate_surrogate(a):
    i, j, v, res, N = a
    
    # generate shuffled residuals
    ndx = np.argsort(np.random.uniform(size = (len(res),)))
    r = np.zeros_like(res)
    r[ndx] = res

    # run the simulation
    return (i, j, v.simulate_with_residuals(r[:N, :]))


class SurrGeoFieldAR(GeoField):
    """Geo field data class that can construct AR(k) models of each time series
       and construct surrogates of itself from the AR(k) models."""
    
    
    def __init__(self):
        """"""
        GeoField.__init__(self)
        self.sd = None
        
        
    def save_field(self, fname):
        """Save the current field in a pickle file.
           The current surrogate data is not saved and must be generated anew after unpickling."""
        with open(fname, "w") as f:
            cPickle.dump([self.d, self.lons, self.lats, self.tm, self.job_data, self.max_ord], f)

        
    def load_field(self, fname):
        with open(fname, "r") as f:
            lst = cPickle.load(f)
            
        self.d = lst[0]
        self.lons = lst[1]
        self.lats = lst[2]
        self.tm = lst[3]
        self.job_data = lst[4]
        self.max_ord = lst[5]


    def surr_data(self):
        """Return the (hopefully already constructed) surrogate data."""
        return self.sd
        

    def copy_field(self, other):
        """Make a deep copy of the data of another Geo Field."""
        self.d = other.d.copy()
        self.lons = other.lons.copy()
        self.lats = other.lats.copy()
        self.tm = other.tm.copy()
        self.job_data = None
            
        
    def prepare_surrogates(self, pool = None):
        """Prepare for generating surrogates by
           (1) identifying the AR model for each time series using sbc criterion,
           (2) compute the residuals which will be shuffled to generate surrogates.
        """

        # optional parallel computation
        if pool == None:
            map_func = map
        else:
            map_func = pool.map
        
        # burst the time series
        num_lats = len(self.lats)
        num_lons = len(self.lons)

        # identify each AR(k) model and compute the residuals
        job_data = [ (i, j, self.d[:, i, j]) 
                     for i in range(num_lats) for j in range(num_lons) ]
        
        job_results = map_func(_prepare_surrogates, job_data)
        
        # find out maximal order which bounds the length of generated surrogates
        max_ord = max([r[2].order() for r in job_results])
        num_tm_s = self.d.shape[0] - max_ord
        
        # create the job_data structure for the surrogate constructor function
        job_data = []
        for i, j, v, r in job_results:
            job_data.append((i, j, v, r, num_tm_s))

        # store both items
        self.max_ord = max_ord
        self.job_data = job_data
        
        return True
    

    def construct_surrogate(self, pool = None):
        """Construct a new surrogate time series."""
        
        # optional parallel computation
        if pool == None:
            map_func = map
        else:
            map_func = pool.map
        
        num_lats = len(self.lats)
        num_lons = len(self.lons)
        num_tm_s = len(self.tm) - self.max_ord
        
        self.sd = np.zeros((num_tm_s, num_lats, num_lons))
        sts_list = map_func(_generate_surrogate, self.job_data)

        for i, j, ts in sts_list:
            self.sd[:, i, j] = ts[:,0]
            
        return True
