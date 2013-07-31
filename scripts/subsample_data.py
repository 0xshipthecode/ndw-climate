
import netCDF4
import sys


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: subsample_data.py <in-file> <var-name> <out-file>")
        sys.exit(1)

    vname = sys.argv[2]
    print("Subsampling variable %s" % vname)
    orig_data = netCDF4.Dataset(sys.argv[1])
    ovars = orig_data.variables

    olat, olon, otime = ovars['lat'][:], ovars['lon'][:], ovars['time']
    odata = ovars[vname]
    
    new_data = netCDF4.Dataset(sys.argv[3], 'w')

    new_data.createDimension('lat', len(olat)//2)
    new_data.createDimension('lon', (len(olon)+1)//2)
    new_data.createDimension('time', None)

    ntime = new_data.createVariable('time','f8',('time',))
    nlat = new_data.createVariable('lat', 'f4', ('lat',))
    nlon = new_data.createVariable('lon', 'f4', ('lon',))
    ndata = new_data.createVariable(vname, odata.dtype, odata.dimensions)
    
    ntime[:] = otime[:]
    nlat[:] = olat[1::2]
    nlon[:] = olon[::2]
    ndata[:] = odata[:][...,1::2,::2]

    orig_data.close()
    new_data.close()
    
    print("Done.")
