

#
#  Module that performs standard data loading.
#

from geo_field import GeoField
from datetime import date


def load_monthly_data_general(fname, varname, from_date, to_date, months, slice_lon, slice_lat, level):
    g = GeoField()
    g.load(fname, varname)
    if level is not None:
        g.slice_level(level)
    g.transform_to_anomalies()
    g.normalize_monthly_variance()
    g.slice_spatial(slice_lon, slice_lat)
    g.slice_date_range(from_date, to_date)
    if months is not None:
        g.slice_months(months)
    return g
    
 
def load_monthly_slp_nh(from_date = date(1948, 1, 1), to_date = date(2012, 1, 1),  months = None):
    return load_monthly_data_general('data/slp.mon.mean.nc', 'slp',
                                     from_date, to_date, months, None, [20, 89],
                                     None)

def load_monthly_slp_all(from_date = date(1948, 1, 1), to_date = date(2012, 1, 1),  months = None):
    return load_monthly_data_general('data/slp.mon.mean.nc', 'slp',
                                     from_date, to_date, months, None, [-89, 89],
                                     None)

def load_monthly_slp_all2(from_date = date(1948, 1, 1), to_date = date(2012, 1, 1),  months = None):
    return load_monthly_data_general('data/slp.mon.mean.nc', 'slp',
                                     from_date, to_date, months, None, None,
                                     None)


def load_monthly_sat_nh(from_date = date(1948, 1, 1), to_date = date(2012, 1, 1),  months = None):
    return load_monthly_data_general('data/air.mon.mean.nc', 'air',
                                     from_date, to_date, months, None, [20, 89], 0)

def load_monthly_sat_all(from_date = date(1948, 1, 1), to_date = date(2012, 1, 1),  months = None):
    return load_monthly_data_general('data/air.mon.mean.nc', 'air',
                                     from_date, to_date, months, None, [-89, 89], 0)

def load_monthly_sat_all2(from_date = date(1948, 1, 1), to_date = date(2012, 1, 1),  months = None):
    return load_monthly_data_general('data/air.mon.mean.nc', 'air',
                                     from_date, to_date, months, None, None, 0)
