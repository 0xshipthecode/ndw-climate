



from datetime import datetime, date
from surr_geo_field_ar import SurrGeoFieldAR
from geo_field import GeoField
from multiprocessing import Pool


# load netCDF SLP field
d = GeoField()
d.load("/home/martin/Work/Geo/data/netcdf/pres.mon.mean.nc", 'pres')
d.slice_date_range(date(1948, 1, 1), date(2012, 1, 1))
#d.slice_months([12, 1, 2])
d.slice_spatial(None, [-89, 89])

# copy into surrogate field
sd = SurrGeoFieldAR()
sd.copy_field(d)

# create the Pool
pool = Pool(4)

t1 = datetime.now()
sd.prepare_surrogates(pool)
print("Prep: elapsed time %s" % str(datetime.now() - t1))

t1 = datetime.now()
sd.construct_surrogate(pool)
print("Gen: elapsed time %s" % str(datetime.now() - t1))

t1 = datetime.now()
sd.construct_surrogate(pool)
print("Gen: elapsed time %s" % str(datetime.now() - t1))