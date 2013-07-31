

#
#  Module that performs standard data loading.
#

from datetime import date
import numpy as np
import cPickle

from geo_field import GeoField


def load_monthly_data_general(fname, varname, from_date, to_date, months, slice_lon, slice_lat, level, var_norm = True):
    g = GeoField()
    g.load(fname, varname)
    if level is not None:
        g.slice_level(level)
    g.transform_to_anomalies()
    if var_norm:
        g.normalize_variance()
    g.slice_spatial(slice_lon, slice_lat)
    g.slice_date_range(from_date, to_date)
    if months is not None:
        g.slice_months(months)
    return g
    
 
def load_monthly_slp_nh(from_date = date(1948, 1, 1),
                        to_date = date(2012, 1, 1),
                        months = None):
    return load_monthly_data_general('data/slp.mon.mean.nc', 'slp',
                                     from_date, to_date, months, None, [20, 89],
                                     None)

def load_monthly_slp_sh(from_date = date(1948, 1, 1),
                        to_date = date(2012, 1, 1),
                        months = None):
    return load_monthly_data_general('data/slp.mon.mean.nc', 'slp',
                                     from_date, to_date, months, 
                                     None, [-89, 0], None)

def load_monthly_slp_all(from_date = date(1948, 1, 1),
                         to_date = date(2012, 1, 1),
                         months = None):
    return load_monthly_data_general('data/slp.mon.mean.nc', 'slp',
                                     from_date, to_date, months, None, [-89, 89],
                                     None)

def load_monthly_slp2x2_all(from_date = date(1948, 1, 1),
                         to_date = date(2012, 1, 1),
                         months = None):
    return load_monthly_data_general('data/slp_2x2.mon.mean.nc', 'slp',
                                     from_date, to_date, months, None, [-89, 89],
                                     None)


def load_monthly_slp_all2(from_date = date(1948, 1, 1),
                          to_date = date(2012, 1, 1),
                          months = None):
    return load_monthly_data_general('data/slp.mon.mean.nc', 'slp',
                                     from_date, to_date, months, None, None,
                                     None)


def load_monthly_sat_nh(from_date = date(1948, 1, 1),
                        to_date = date(2012, 1, 1),
                        months = None):
    return load_monthly_data_general('data/air.mon.mean.nc', 'air',
                                     from_date, to_date, months, None, [20, 89], 0)

def load_monthly_sat_sh(from_date = date(1948, 1, 1),
                        to_date = date(2012, 1, 1),
                        months = None):
    return load_monthly_data_general('data/air.mon.mean.nc', 'air',
                                     from_date, to_date, months, None, [-89, 0], 0)

def load_monthly_sat_all(from_date = date(1948, 1, 1),
                         to_date = date(2012, 1, 1),
                         months = None):
    return load_monthly_data_general('data/air.mon.mean.nc', 'air',
                                     from_date, to_date, months, None, [-89, 89], 0)

def load_monthly_sat_all2(from_date = date(1948, 1, 1),
                          to_date = date(2012, 1, 1),
                          months = None):
    return load_monthly_data_general('data/air.mon.mean.nc', 'air',
                                     from_date, to_date, months, None, None, 0)


def load_monthly_hgt500_all(from_date = date(1948, 1, 1),
                          to_date = date(2012, 1, 1),
                          months = None):
    return load_monthly_data_general("data/hgt.mon.mean.nc", "hgt",
                                     from_date, to_date, months, None, [-89, 89], 0)



def load_daily_data_general(fname, varname, from_date, to_date,
                            slice_lon, slice_lat, level):

    # the daily data is stored in yearly files
    yr_start = from_date.year
    yr_end = to_date.year

    # load each NC dataset
    gflist = []
    Ndays = 0
    for yr in range(yr_start, yr_end+1):
        g = GeoField()
        g.load(fname % yr, varname)
        if level is not None:
            g.slice_level(level)
        g.slice_spatial(slice_lon, slice_lat)
        g.slice_date_range(from_date, to_date)
        Ndays += len(g.tm)
        gflist.append(g)

    # now append all of the records together
    g = GeoField()
    d = np.zeros((Ndays, len(gflist[0].lats), len(gflist[0].lons)))
    tm = np.zeros((Ndays,))
    n = 0
    for g_i in gflist:
        Ndays_i = len(g_i.tm)
        d[n:Ndays_i + n, :, :] = g_i.d
        tm[n:Ndays_i + n] = g_i.tm
        n += Ndays_i

    # load geo_fields and then append them together
    g.use_existing(d, gflist[0].lons, gflist[0].lats, tm)
    g.transform_to_anomalies()
    g.normalize_variance()

    return g


def load_daily_sat_all2(from_date = date(1948, 1, 1),
                        to_date = date(2012, 1, 1)):
    """
    Loads daily SAT data without slicing poles away and keeping all
    days of the year from pole to pole (all).  Data is loaded from
    air.[year].nc files (only level 0 is used).
    """

    return load_daily_data_general('data/air.%d.nc', 'air',   # file
                                   from_date, to_date,        # date range
                                   None, None, 0)


def load_daily_slp_all2(from_date = date(1948, 1, 1),
                        to_date = date(2012, 1, 1)):
    """
    Loads daily SLP data without slicing poles away and keeping all
    days of the year from pole to pole (all).
    """

    return load_daily_data_general('data/slp.%d.nc', 'slp',   # file
                                   from_date, to_date,        # date range
                                   None, None, None)


def load_daily_slp_sh(from_date = date(1948, 1, 1),
                      to_date = date(2012, 1, 1)):
    """
    Loads daily SLP data without slicing poles away and keeping all
    days of the year from pole to pole (all).
    """

    return load_daily_data_general('data/slp.%d.nc', 'slp',   # file
                                   from_date, to_date,        # date range
                                   None, [-89, 0], None)




def load_daily_sat_sh(from_date = date(1948, 1, 1),
                      to_date = date(2012, 1, 1)):
    """
    Loads daily SAT data for the southern hemisphere (-89, 0) LAT,
    days of the year from pole to pole (all).  Data is loaded from
    air.[year].nc files (only level 0 is used).
    """

    return load_daily_data_general('data/air.%d.nc', 'air',   # file
                                   from_date, to_date,        # date range
                                   None, [-89, 0], 0)


def load_dumped_objects(fname):
    """
    Load a set of dumped objects using cPickle and return it.
    """
    with open(fname, 'r') as f:
        return cPickle.load(f)
