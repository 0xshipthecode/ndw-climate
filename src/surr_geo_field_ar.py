# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:02:39 2012

@author: martin
"""


from geo_field import GeoField
from var_model import VARModel
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
    return (i, j, v.simulate_with_residuals(r[:N, np.newaxis]))


class SurrGeoFieldAR(GeoField):
    """Geo field data class that can construct AR(k) models of each time series
       and construct surrogates of itself from the AR(k) models."""
    
    
    def __init__(self):
        """"""
        GeoField.__init__(self)
        

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
        
        # model_grid and residuals are both dicts
        self.residuals = {}
        self.model_grid = {}

        job_data = [ (i, j, self.d[:, i, j]) 
                     for i in range(num_lats) for j in range(num_lons) ]
        
        job_results = map_func(_prepare_surrogates, job_data)
        
        # store the job results in the object
        max_ord = 0
        for i, j, v, r in job_results:
            self.model_grid[(i,j)] = v
            max_ord = max(max_ord, v.order())
            self.residuals[(i,j)] = r[:, 0]

        self.max_ord = max_ord
        return True
    

    def construct_surrogate(self, pool = None):
        """Construct a surrogate in-place."""
        
        # optional parallel computation
        if pool == None:
            map_func = map
        else:
            map_func = pool.map
        
        num_lats = len(self.lats)
        num_lons = len(self.lons)
        num_tm = len(self.tm)
        
        num_tm_s = num_tm - self.max_ord
        
        self.sd = np.zeros((num_tm_s, num_lats, num_lons))
        
        if self.job_data == None:
            self.job_data = [ (i, j, self.model_grid[(i,j)], self.residuals[(i,j)], num_tm_s)
                              for i in range(num_lats) for j in range(num_lons)]
        
        sts_list = map_func(_generate_surrogate, self.job_data)

        for i, j, ts in sts_list:
            self.sd[:, i, j] = ts[:,0]
            
        return True
