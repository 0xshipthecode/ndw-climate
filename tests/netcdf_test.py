# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:17:39 2012

@author: martin
"""

from datetime import date
from geo_field import GeoField
from var_model import VARModel


d = GeoField()
d.load("/home/martin/Work/Geo/data/netcdf/pres.mon.mean.nc", 'pres')
d.remove_anomalies()
d.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
#d.slice_months([12, 1, 2])
d.slice_spatial(None, [-89, 89])




