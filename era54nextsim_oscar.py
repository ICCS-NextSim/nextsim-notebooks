#script to convert new ERA5 files to readable nextsim ERA5 files 
# main chage: convert `valid_time` into `time` and a few file and variable names
 
print('On nesi use with: load-python-nextsim')
print('You need to close the ncfile "ncfile.close()" before opening it. ')

import xarray as xr
import numpy as np
#import matplotlib.pyplot as plt
import numpy.ma as ma
#import cmocean
#from matplotlib.animation import FuncAnimation
#from matplotlib import animation, rc
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4 as nc
from netCDF4 import Dataset
import datetime as dt
from netCDF4 import date2num,num2date
from sys import exit
import socket
import os
#plt.ion()
#plt.close('all')


firstYear = 2018 # int(sys.argv[1])
lastYear  = 2018 # int(sys.argv[2])

files_in=[
##'d2m',
##'avg_sdlwrf',
##'avg_sdswrf',
#'msl',
#'avg_tsrwe',
'avg_tprate',
##'t2m', 
##'u10', 
#'v10'
] 

files_out=[
##'d2m',
##'msdwlwrf',
##'msdwswrf',
#'msl',
#'msr',
'mtpr',
##'t2m',
##'u10',
#'v10',
]

print('Hostname: '+socket.gethostname())
if socket.gethostname()[-11::]=='nesi.org.nz':
  path_in='/nesi/project/uoa03669/data/ERA5/new_files/' 
  path_out='/nesi/project/uoa03669/data/ERA5/proc/' 
#elif socket.gethostname()[-11::]=='nesi.org.nz': # oscar
  #path_in='/oscar/data/deeps/private/chorvat/nextsim/ERA5/'
  #path_out='/oscar/data/deeps/private/chorvat/nextsim/ERA5/proc/'
else:
  print("Your input and output paths haven't been set for your computing enviroment")
  exit()

for year in range(firstYear,lastYear+1):
    k=-1
    for f_in in files_in:
        k+=1
        f_out=files_out[k] # f_in and f_out are use to read the nc file

        file_in=path_in+'ERA5_'+f_in+'_y'+str(year)+'.nc'
        file_out=path_out+'ERA5_'+f_out+'_y'+str(year)+'.nc'
        print('\nProcessing file '+file_in)

        # reading sose lon,lat,time
        ds=Dataset(file_in,'r')
        lon_in=ds.variables['longitude'][:] 
        lat_in=ds.variables['latitude'][:] 
        time_in=ds.variables['valid_time'][:] 
        data_in=ds.variables[f_in][:,:,:]
        ds.close()

        # opening/creating file for writing
        #if ncfile._isopen==1:
        #  ncfile.close()  # just to be safe, make sure dataset is not already open.
        if os.path.exists(file_out):
            os.remove(file_out)
            print(f"Deleted old file: {file_out}")
        print('\nCreating file '+file_out)
        #ncfile = Dataset(path+'BSOSE_I139_2013to2021_5d_'+file_type+'.nc',mode='w',format='NETCDF4_CLASSIC') 
        ncfile = Dataset(file_out,mode='w',format='NETCDF4_CLASSIC') 
        print(ncfile)
        #exit()

        lon_dim = ncfile.createDimension('longitude', len(lon_in))    # longitude axis
        lat_dim = ncfile.createDimension('latitude', len(lat_in)) # ilat_sose)+len(lat_sose))     # latitude axis
        #depth_dim = ncfile.createDimension('depth', 1) # 
        time_dim = ncfile.createDimension('time', len(time_in)) # unlimited axis = None or 0
        for dim in ncfile.dimensions.items():
            print(dim)
        ncfile.title='ERA5 file processed for nextsim. `valid_time` changed to `time` by `era54nextsim.py`'
        #ncfile.set_fill_off() #ncattr("_Fill_Value", None)
        
        # Define two variables with the same names as dimensions,
        # a conventional way to define "coordinate variables".
        longitude = ncfile.createVariable('longitude', np.float32, ('longitude',))
        longitude.units = 'degrees_east'
        longitude.long_name = 'Longitude'
        latitude = ncfile.createVariable('latitude', np.float32, ('latitude',))
        latitude.units = 'degrees_north'
        latitude.long_name = 'Latitude'
        #depth = ncfile.createVariable('depth', np.float64, ('depth',))
        #depth.units = "m"
        #depth.long_name = 'Depth'
        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = "seconds since 1970-01-01"
        #time.units = "hours since 1950-01-01 00:00:00"
        time.long_name = 'time'
        
        # Write time, latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        #li=len(ilat_sose)
        #ll=len(ilat_sose)+len(lat_sose)
        #latitude[0:li] = ilat_sose # -90. + (180./nlats)*np.arange(nlats) # south pole to north pole
        #latitude[li:ll] = lat_sose # -90. + (180./nlats)*np.arange(nlats) # south pole to north pole
        latitude[:] = lat_in # (180./nlats)*np.arange(nlons) # Greenwich meridian eastward
        longitude[:] = lon_in # (180./nlats)*np.arange(nlons) # Greenwich meridian eastward
        time[:] = time_in #

        #exit()

        # reading original bsose files
        # and writing new nc files based on GLORYS file format
        #files={'SSH_bsoseI139_2013to2021_5dy.nc','Uvel_bsoseI139_2013to2021_5dy.nc_26m','Vvel_bsoseI139_2013to2021_5dy.nc_26m'}
        # loop in the variables
        #for f in files:
        #f=file_out

        #ds=Dataset(f,'r')
        #data_in=ds.variables[f_in][:,:,:] 
        #ds.close()
        # replacing fillvalue by a fixed one
        #im=ma.getmaskarray(sose);
        #im = np.where(mask)[0]
        #im=sose==0.; sose[im]=-0.860163 
        #im=sose>=2.; sose[im]=2. 

        # Define a 3D variable to hold the data
        if f_out=='d2m':
            d2m = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            d2m.long_name = "2 metre dewpoint temperature" ;
            d2m.units = "K" ;
            d2m.add_offset = 247.095742485891 ;
            d2m.scale_factor = 0.0017675916949911 ;
            d2m[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='msdwswrf':
            data = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            #data.setncattr("missing_value", np.int16(-32767))
            data.long_name = 'Mean surface downward short-wave radiation flux' # this is a CF standard name
            data.units = "W m**-2" ;
            data.add_offset = 647.677616620634 ;
            data.scale_factor = 0.0197667587322418 ;
            #data._FillValue = -32767s ;
            #data.missing_value = -32767s ;
            data[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='msdwlwrf':
            data = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            #data.setncattr("missing_value", np.int16(-32767))
            data.long_name = 'Mean surface downward long-wave radiation flux' 
            data.units = "W m**-2" ;
            data.add_offset = 288.874343441547 ;
            data.scale_factor = 0.00749292647682334 ;
            #data._FillValue = -32767s ;
            #data.missing_value = -32767s ;
            data[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='msl':
            msl = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            msl.standard_name = "air_pressure_at_mean_sea_level" ;
            msl.long_name = "Mean sea level pressure" ;
            msl.units = "Pa" ;
            msl.add_offset = 99233.1079646896 ;
            msl.scale_factor = 0.22157062090855 ;
            msl[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='msr':
            msr = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            msr.long_name = "Mean snowfall rate" ;
            msr.units = "kg m**-2 s**-1" ;
            msr.add_offset = 0.00122688328616068 ;
            msr.scale_factor = 3.74437919233559e-08 ;
            msr[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='mtpr':
            mtpr = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            mtpr.long_name = "Mean total precipitation rate" ;
            mtpr.units = "kg m**-2 s**-1" ;
            mtpr.add_offset = 0.0103438706283263 ;
            mtpr.scale_factor = 3.15689148151325e-07 ;
            mtpr[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='t2m':
            t2m = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            t2m:long_name = "2 metre temperature" ;
            t2m:units = "K" ;
            t2m:add_offset = 258.443991925885 ;
            t2m:scale_factor = 0.00201553787823561 ;
            t2m[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='u10':
            u10 = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            u10.long_name = "10 metre U wind component" ;
            u10.units = "m s**-1" ;
            u10.add_offset = -11.9843451655783 ;
            u10.scale_factor = 0.0014509512736993 ;
            u10[:,:,:]=data_in
            ncfile.set_fill_off()

        elif f_out=='v10':
            v10 = ncfile.createVariable(f_out,np.float64,('time','latitude','longitude'))#,fill_value=-32767) # note: unlimited dimension is leftmost
            v10.long_name = "10 metre V wind component" ;
            v10.units = "m s**-1" ;
            v10.add_offset = 11.1050833641156 ;
            v10.scale_factor = 0.00153371122197619 ;
            v10[:,:,:]=data_in
            ncfile.set_fill_off()

        #data[:,0:li,:]=-0.860163
        #data[:,li:ll,:]=sose
        #ncfile.set_fill_off() #ncattr("_Fill_Value", None)
        #zos=np.ma.filled(zos.astype(float), -0.860163)

print('  ')
print('Compress netcdf with:')
print('ml NCO')
print('load-miniforge')
print('conda activate /nesi/project/uoa03669/rsan613/envs/era5')
print('nccopy -d9 -s ERA5_msdwlwrf_y2018.nc ERA5_msdwlwrf_y2018.nc4')

exit()

