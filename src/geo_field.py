# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:17:39 2012

@author: martin
"""

from datetime import date
import numpy as np
from netCDF4 import Dataset


class GeoField:
    """A class that represents the time series of a geographic field.  All represented fields
       have two spatial dimensions (longitude, latitude) and one temporal dimension.  Optionally
       the fields may have a height dimension.""" 
    
    def __init__(self):
        """Initialize to empty data."""

        self.d = None
        self.lons = None
        self.lats = None
        self.tm = None
        
    def data(self):
        return self.d
        
    
    def use_existing(self, d, lons, lats, tm):
        """Initialize with already existing data."""

        self.d = d
        self.lons = lons
        self.lats = lats
        self.tm = tm
        
        
    def load(self, netcdf_file, var_name):
        """Load GeoData structure from netCDF file."""

        d = Dataset(netcdf_file)
        v = d.variables[var_name]
        
        # extract the data
        self.d = v[:]
        
        # extract spatial & temporal info
        self.lons = d.variables['lon'][:]
        self.lats = d.variables['lat'][:]
        self.tm = d.variables['time'][:] / 24.0

        d.close()
        

    def slice_date_range(self, date_from, date_to):
        """Subselects the date range.  Date_from is inclusive, date_to is not.
           Modification is in-place due to volume of data."""
        
        d_start = date_from.toordinal()
        d_stop = date_to.toordinal()
        
        ndx = np.logical_and(self.tm >= d_start, self.tm < d_stop)
        self.tm = self.tm[ndx]
        self.d = self.d[ndx, ...]
        
        
    def slice_months(self, months):
        """Subselect only certain months, not super efficient but useable, since
           upper bound on len(months) = 12.
           Modification is in-place due to volume of data."""

        tm = self.tm
        ndx = filter(lambda i: date.fromordinal(int(tm[i])).month in months, range(len(tm)))
        
        self.tm = tm[ndx]
        self.d = self.d[ndx, ...]
        

    def slice_spatial(self, lons, lats):
        """Slice longitude and/or lattitude.  None means don't modify dimension.
           Both arguments are ranges [from, to], both limits are inclusive."""
           
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
        
