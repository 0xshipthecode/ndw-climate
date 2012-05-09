"""
Created on Thu Mar  1 11:17:39 2012

@author: martin
"""

import numpy as np
import scipy as sp
import math
from datetime import date
from netCDF4 import Dataset


class GeoField:
    """
    A class that represents the time series of a geographic field.  All represented fields
    have two spatial dimensions (longitude, latitude) and one temporal dimension.  Optionally
    the fields may have a height dimension.
    """ 
    
    def __init__(self):
        """
        Initialize to empty data.
        """
        self.d = None
        self.lons = None
        self.lats = None
        self.tm = None
        self.cos_weights = None
        
    
    def data(self):
        """
        Return a copy of the stored data as a multi-array.
        If access without copying is required for performance reasons, use self.d at your own risk.
        """
        return self.d.copy()
        
    
    def use_existing(self, d, lons, lats, tm):
        """
        Initialize with already existing data.
        """
        self.d = d
        self.lons = lons
        self.lats = lats
        self.tm = tm
        
        
    def load(self, netcdf_file, var_name):
        """
        Load GeoData structure from netCDF file.
        """
        d = Dataset(netcdf_file)
        v = d.variables[var_name]
        
        # extract the data
        self.d = v[:]
        
        # extract spatial & temporal info
        self.lons = d.variables['lon'][:]
        self.lats = d.variables['lat'][:]
        self.tm = d.variables['time'][:] / 24.0

        d.close()
        

    def slice_level(self, lev):
        """
        Slice the data and keep only one level (presupposes that the data has levels).
        """
        if self.d.ndim > 3:
            self.d = self.d[:, lev, ...]
        else:
            raise "Slicing levels in single-level data!"
    
    def slice_date_range(self, date_from, date_to):
        """
        Subselects the date range.  Date_from is inclusive, date_to is not.
        Modification is in-place due to volume of data.
        """
        
        d_start = date_from.toordinal()
        d_stop = date_to.toordinal()
        
        ndx = np.logical_and(self.tm >= d_start, self.tm < d_stop)
        self.tm = self.tm[ndx]
        self.d = self.d[ndx, ...]
        

    def find_date_ndx(self, dt):
        """
        Return the index that corresponds to the specific day.
        Returns none if the date is not contained in the data.
        """
        d_day = dt.toordinal()
        pos = np.nonzero(self.tm == d_day)
        if len(pos) == 1:
            return pos[0]
        else:
            return None
        
        
    def slice_months(self, months):
        """
        Subselect only certain months, not super efficient but useable, since upper bound
        on len(months) = 12. Modification is in-place due to volume of data.
        """
        tm = self.tm
        ndx = filter(lambda i: date.fromordinal(int(tm[i])).month in months, range(len(tm)))
        
        self.tm = tm[ndx]
        self.d = self.d[ndx, ...]
        

    def slice_spatial(self, lons, lats):
        """
        Slice longitude and/or latitude.  None means don't modify dimension.
        Both arguments are ranges [from, to], both limits are inclusive.
        """
        if lons != None:
            lon_ndx = np.nonzero(np.logical_and(self.lons >= lons[0], self.lons <= lons[1]))[0]
        else:
            lon_ndx = np.arange(len(self.lons))
            
        if lats != None:
            lat_ndx = np.nonzero(np.logical_and(self.lats >= lats[0], self.lats <= lats[1]))[0]
        else:
            lat_ndx = np.arange(len(self.lats))

        # apply slice to the data (by dimensions, as slicing in two dims at the same time doesn't work)             
        d = self.d
        d = d[..., lat_ndx, :]
        self.d = d[..., lon_ndx]
            
        # apply slice to the vars
        self.lons = self.lons[lon_ndx]
        self.lats = self.lats[lat_ndx]
        

    def transform_to_anomalies(self):
        """
        Remove the yearly cycle from the time series.
        """
        # check if data is monthly or daily
        d = self.d
        delta = self.tm[1] - self.tm[0]
        
        if abs(delta - 1.0) < 0.1:
            # daily data
            for i in range(365):
                mn = np.mean(d[i::365, :, :], axis = 0)
                d[i::365, :, :] -= mn
        elif abs(delta - 30) < 3.0:
            # monthly data
            for i in range(12):
                mn = np.mean(d[i::12, :, :], axis = 0)
                d[i::12, :, :] -= mn
        else:
            raise "Unknown temporal sampling in geographical field."


    def normalize_monthly_variance(self):
        """
        Normalize the variance of monthly values.
        """
        # check if data is monthly
        d = self.d
        delta = self.tm[1] - self.tm[0]

        if abs(delta - 30) < 3.0:
            # monthly data
            for i in range(12):
                mn = np.std(d[i::12, :, :], axis = 0)
                d[i::12, :, :] /= mn
        else:
            raise "Not monthly data!"

    
    def sample_temporal_bootstrap(self):
        """
        Return a temporal bootstrap sample.
        """
        # select samples at random
        num_tm = len(self.tm)
        ndx = sp.random.random_integers(0, num_tm - 1, size = (num_tm,))
        
        # return a copy of the resampled dataset
        return self.d[ndx, ...].copy()
    
    
    def spatial_dims(self):
        """
        Return the spatial dimensions of the data.  Spatial dimensions are all dimensions
        after the first one, which corresponds to time.
        """
        return self.d.shape[1:]
    
    
    def reshape_flat_field(self, f):
        """
        Reshape a field that has been spatially flattened back into shape that
        corresponds to the spatial dimensions of this field.  It is assumed that
        f is a 2D array with axis 0 spatial and axis 1 temporal.  The matrix will
        be transposed and the spatial dimension reshaped back to match the spatial
        dimensions of the data in this field.
        """
        n = f.shape[1]
        f = f.transpose()
        new_shape = [n] + list(self.spatial_dims())
        return f.reshape(new_shape)
    
    
    def qea_latitude_weights(self):
        """
        Return a grid which contains the scaling factors to rescale each time series
        by sqrt(cos(lattitude)).
        """
        # if cached, return cached version
        if self.cos_weights != None:
            return self.cos_weights
        
        # if not cached recompute, cache and return
        cos_weights = np.zeros_like(self.d[0, :, :])
        lats = self.lats
        for lat_ndx in range(len(lats)):
                cos_weights[lat_ndx, :] = math.cos(lats[lat_ndx] * math.pi / 180) ** 0.5
                 
        self.cos_weights = cos_weights
        return cos_weights
