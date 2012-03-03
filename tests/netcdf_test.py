# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:17:39 2012

@author: martin
"""

from datetime import date
from geo_data import GeoData
from var_model import VarModel


d = GeoData()
d.load("/home/martin/Work/Geo/data/netcdf/pres.mon.mean.nc", 'pres')
d.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
#d.slice_months([12, 1, 2])
d.slice_spatial(None, [-89, 89])




