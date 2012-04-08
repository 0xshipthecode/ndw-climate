# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:17:39 2012

@author: martin
"""

from datetime import date
from geo_field import GeoField
from var_model import VARModel
from geo_rendering import render_component_single
import matplotlib.pyplot as plt

d = GeoField()
d.load("data/pres.mon.mean.nc", 'pres')
d.transform_to_anomalies()
d.normalize_monthly_variance()
d.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
#d.slice_months([12, 1, 2])
d.slice_spatial(None, [-89, 89])


render_component_single(d.d[0, :, :], d.lats, d.lons, False, None, 'SLP anomalies Jan 1948')

render_component_single(d.d[-1, :, :], d.lats, d.lons, False, None, 'SLP anomalies Jan 2012')
plt.show()
