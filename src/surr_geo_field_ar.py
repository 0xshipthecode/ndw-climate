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
    i, j, order_range, crit, ts = a
    v = VARModel()
    v.estimate(ts, order_range, True, crit, None)
    r = v.compute_residuals(ts)
    return(i, j, v, r)


class SurrGeoFieldAR(GeoField):
    """Geo field data class that can construct AR(k) models of each time series
       and construct surrogates of itself from the AR(k) models."""
    
    
    def __init__(self, order_range = [0, 30], crit = 'sbc'):
        """"""
        GeoField.__init__(self)
        self.sd = None
        self.order_range = order_range
        self.crit = crit
        
        
    def save_field(self, fname):
        """Save the current field in a pickle file.
           The current surrogate data is not saved and must be generated anew after unpickling."""
        with open(fname, "w") as f:
            cPickle.dump([self.d, self.lons, self.lats, self.tm, self.max_ord, self.model_grid, self.residuals], f)

        
    def load_field(self, fname):
        with open(fname, "r") as f:
            lst = cPickle.load(f)
            
        self.d = lst[0]
        self.lons = lst[1]
        self.lats = lst[2]
        self.tm = lst[3]
        self.max_ord = lst[4]
        self.model_grid = lst[5]
        self.residuals = lst[6]


    def surr_data(self):
        """Return the (hopefully already constructed) surrogate data."""
        return self.sd
        

    def copy_field(self, other):
        """Make a deep copy of the data of another Geo Field."""
        self.d = other.d.copy()
        self.lons = other.lons.copy()
        self.lats = other.lats.copy()
        self.tm = other.tm.copy()
            
        
    def prepare_surrogates(self, pool = None):
        """Prepare for generating surrogates by
           (1) identifying the AR model for each time series using sbc criterion,
           (2) compute the residuals which will be shuffled to generate surrogates.
        """
        
        if pool == None:
            map_func = map
        else:
            map_func = pool.map
        
        # burst the time series
        num_lats = len(self.lats)
        num_lons = len(self.lons)
        num_tm = len(self.tm)

        # run the job in parallel
        job_data = [ (i, j, self.order_range, self.crit, self.d[:, i, j]) for i in range(num_lats) for j in range(num_lons)]
        job_results = map_func(_prepare_surrogates, job_data)
        
        # find out maximal order (and thus length of residuals for entire dataset)
        max_ord = max([r[2].order() for r in job_results])
        num_tm_s = num_tm - max_ord
        
        self.model_grid = np.zeros((num_lats, num_lons), dtype = np.object)
        self.residuals = np.zeros((num_tm_s, num_lats, num_lons), dtype = np.float64)

        for i, j, v, r in job_results:
            self.model_grid[i,j] = v
            self.residuals[:, i, j] = r[:num_tm_s, 0]

        # store both items
        self.max_ord = max_ord
    

    def construct_surrogate_with_residuals(self):
        """Construct a new surrogate time series.  The construction is not done in parallel as
           the entire surrogate generation and processing loop will be split into independent tasks.
        """
        
        num_lats = len(self.lats)
        num_lons = len(self.lons)
        num_tm_s = len(self.tm) - self.max_ord
        
        self.sd = np.zeros((num_tm_s, num_lats, num_lons))

        # space for shuffled residuals
        r = np.zeros((num_tm_s,1), dtype = np.float64)
        
        # run through entire grid
        for i in range(num_lats):
            for j in range(num_lons):
                # generate shuffled residuals
                ndx = np.argsort(np.random.uniform(size = (num_tm_s,)))
                r[ndx, 0] = self.residuals[:, i, j]

                self.sd[:, i, j] = self.model_grid[i, j].simulate_with_residuals(r)[:, 0]


    def construct_surrogate_with_noise(self):
        """
        Construct a new surrogate time series.  The construction is not done in parallel as
        the entire surrogate generation and processing loop will be split into independent tasks.
        The AR processes will be fed noise according to the noise covariance matrix.  100 samples
        will be used to spin-up the models.
        """
        
        num_lats = len(self.lats)
        num_lons = len(self.lons)
        num_tm = len(self.tm)
        
        self.sd = np.zeros((num_tm, num_lats, num_lons))

        # run through entire grid
        for i in range(num_lats):
            for j in range(num_lons):
                self.sd[:, i, j] = self.model_grid[i, j].simulate(num_tm)[:,0]


    def construct_white_noise_surrogates(self):
        """
        Construct white-noise (shuffling) surrogates.
        """
        num_lats = len(self.lats)
        num_lons = len(self.lons)
        self.sd = np.zeros_like(self.d)
        
        for i in range(num_lats):
            for j in range(num_lons):
                self.sd[:, i, j] = np.random.permutation(self.d[:, i, j])
                
                
    def construct_fourier1_surrogates(self):
        """
        Construct Fourier type-1 surrogates (independent realizations in each
        time series).
        """
        xf = np.fft.rfft(self.d, axis = 0)

        # generate uniformely distributed
        angle = np.random.uniform(0, 2 * np.pi, xf.shape)

        # for 0 rotation for constant elements
        angle[0, :] = 0

        # generate a copy and rotate randomly
        cxf = xf * np.exp(complex(0,1) * angle)

        # inverse real FFT
        self.sd = np.fft.irfft(cxf, axis = 0)
        

    
    def model_orders(self):
        """Return model orders of all models in grid."""
        mo = np.zeros_like(self.model_grid, dtype = np.int32)
        
        for i in range(len(self.lats)):
            for j in range(len(self.lons)):
                mo[i,j] = self.model_grid[i, j].order()
                
        return mo
