# processes BSOSE 5 day averages of SSH, ICE conc and thick, MLD, and sliced U,V,T,S
# download BSOSE files on http://sose.ucsd.edu/SO6/ITER139/  , using: 
# curl -C - -O http://sose.ucsd.edu/SO6/ITER139/SeaIceArea_bsoseI139_2013to2021_5dy.nc
# slice with ncea, e.g.:
# ncea -O -F -d Z,5 Uvel_bsoseI139_2013to2021_5dy.nc Uvel_bsoseI139_2013to2021_5dy.nc_26m

#Input
#########################################################################
file_type = '3m'  # file_type='3m' or '30m'
ini_lat   = -81.00003 # initial glorys latitude
#########################################################################

print('You need to close the ncfile "ncfile.close()" before opening it. ')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4 as nc
from netCDF4 import Dataset
import datetime as dt
from netCDF4 import date2num,num2date
from sys import exit
plt.ion()
plt.close('all')

# reading sose lon,lat,time
path='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/BSOSE/'
ds=Dataset(path+'SSH_bsoseI139_2013to2021_5dy.nc','r')
lon_sose=ds.variables['XC'][:] 
lat_sose=ds.variables['YC'][:] 
time_in=ds.variables['time'][:] 
ds.close()

# making long=0:360 for nextsim
lon_sose[0]=0.; lon_sose[-1]=360.

# extending sose (dummy values) south
ilat_sose=np.arange(ini_lat,lat_sose[0],lat_sose[1]-lat_sose[0])

#modifying sose time
time_date=num2date(time_in,"seconds since 2012-12-01")
time_out=date2num(time_date,"hours since 1950-01-01 00:00:00")

# opening/creating file for writing
#if ncfile._isopen==1:
#  ncfile.close()  # just to be safe, make sure dataset is not already open.
path_out='/Users/rsan613/data/'
#ncfile = Dataset(path+'BSOSE_I139_2013to2021_5d_'+file_type+'.nc',mode='w',format='NETCDF4_CLASSIC') 
ncfile = Dataset(path_out+'GLORYS12V1_2018_'+file_type+'.nc',mode='w',format='NETCDF4_CLASSIC') 
print(ncfile)
lon_dim = ncfile.createDimension('longitude', len(lon_sose))    # longitude axis
lat_dim = ncfile.createDimension('latitude', len(ilat_sose)+len(lat_sose))     # latitude axis
depth_dim = ncfile.createDimension('depth', 1) # 
time_dim = ncfile.createDimension('time', len(time_out)) # unlimited axis = None or 0
for dim in ncfile.dimensions.items():
    print(dim)
ncfile.title='BSOSE (http://sose.ucsd.edu/) prescribed ocean forcing for neXtSIM standalone, `bsose4nextsim.py`, R. Santana, 2022.'
ncfile.set_fill_off() #ncattr("_Fill_Value", None)

# Define two variables with the same names as dimensions,
# a conventional way to define "coordinate variables".
longitude = ncfile.createVariable('longitude', np.float32, ('longitude',))
longitude.units = 'degrees_east'
longitude.long_name = 'Longitude'
latitude = ncfile.createVariable('latitude', np.float32, ('latitude',))
latitude.units = 'degrees_north'
latitude.long_name = 'Latitude'
depth = ncfile.createVariable('depth', np.float64, ('depth',))
depth.units = "m"
depth.long_name = 'Depth'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = "hours since 1950-01-01 00:00:00"
time.long_name = 'time'

# Write time, latitudes, longitudes.
# Note: the ":" is necessary in these "write" statements
li=len(ilat_sose)
ll=len(ilat_sose)+len(lat_sose)
latitude[0:li] = ilat_sose # -90. + (180./nlats)*np.arange(nlats) # south pole to north pole
latitude[li:ll] = lat_sose # -90. + (180./nlats)*np.arange(nlats) # south pole to north pole
longitude[:] = lon_sose # (180./nlats)*np.arange(nlons) # Greenwich meridian eastward
time[:] = time_out #

# writing 30m (U,V,SSH) or 3m (T,S,MLD,ICE conc, ICE thick) 
if file_type=='30m':

  # reading original bsose files
  # and writing new nc files based on GLORYS file format
  files={'SSH_bsoseI139_2013to2021_5dy.nc','Uvel_bsoseI139_2013to2021_5dy.nc_26m','Vvel_bsoseI139_2013to2021_5dy.nc_26m'}
  # loop in the variables
  for f in files:
    print('') 
    print('Reading: '+f) 

    ds=Dataset(path+f,'r')
    if f=='SSH_bsoseI139_2013to2021_5dy.nc':
      sose=ds.variables['ETAN'][:,:,:] 
      ds.close()
      # add ice thickness to SSH
      ds=Dataset(path+'SeaIceHeff_bsoseI139_2013to2021_5dy.nc','r')
      ice_thick=ds.variables['SIheff'][:,:,:] 
      ds.close()
      # ntotal=n+ice_thick*(rho_ice/rho_ocean); rho_ice=934; rho_ocean=1025;     
      sose = sose + ice_thick*(934./1025.)
      # replacing fillvalue by a fixed one
      #im=ma.getmaskarray(sose);
      #im = np.where(mask)[0]
      im=sose==0.; sose[im]=-0.860163 
      im=sose>=2.; sose[im]=2. 

      # Define a 3D variable to hold the data
      zos = ncfile.createVariable('zos',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
      zos.units = 'm' # degrees Kelvin
      zos.long_name = 'Sea surface height' # this is a CF standard name
      print(zos)
      zos[:,0:li,:]=-0.860163
      zos[:,li:ll,:]=sose
      ncfile.set_fill_off() #ncattr("_Fill_Value", None)
      #zos=np.ma.filled(zos.astype(float), -0.860163)
    elif f=='Uvel_bsoseI139_2013to2021_5dy.nc_26m':
      sose=ds.variables['Z'][:] 
      depth[:]=np.abs(sose)
      sose=ds.variables['UVEL'][:,:,:,:] 
      # replacing fillvalue by a fixed one
      im=ma.getmaskarray(sose)
      im=sose==0.; sose[im]=0. 
      ds.close()
      # Define a 3D variable to hold the data
      uo = ncfile.createVariable('uo',np.float64,('time','depth','latitude','longitude')) # note: unlimited dimension is leftmost
      uo.units = 'm s-1' # degrees Kelvin
      uo.long_name = 'Eastward velocity' 
      print(uo)
      uo[:,:,0:li,:]=0.
      uo[:,:,li:ll,:]=sose
    elif f=='Vvel_bsoseI139_2013to2021_5dy.nc_26m':
      sose=ds.variables['VVEL'][:,:,:,:] 
      # replacing fillvalue by a fixed one
      im=ma.getmaskarray(sose)
      im=sose==0.; sose[im]=0. 
      ds.close()
      # Define a 3D variable to hold the data
      vo = ncfile.createVariable('vo',np.float64,('time','depth','latitude','longitude')) # note: unlimited dimension is leftmost
      vo.units = 'm s-1' # degrees Kelvin
      vo.long_name = 'Northward velocity' 
      print(vo)
      vo[:,:,0:li,:]=0.
      vo[:,:,li:ll,:]=sose

# writing 3m (T,S,MLD,ICE conc, ICE thick) 
elif file_type=='3m':

  # reading original bsose files
  # and writing new nc files based on GLORYS file format
  files={'Theta_bsoseI139_2013to2021_5dy.nc_2m','Salt_bsoseI139_2013to2021_5dy.nc_2m','MLD_bsoseI139_2013to2021_5dy.nc',
         'SeaIceArea_bsoseI139_2013to2021_5dy.nc','SeaIceHeff_bsoseI139_2013to2021_5dy.nc'}
  #files={'MLD_bsoseI139_2013to2021_5dy.nc'}

  # loop in the variables
  for f in files:
    
    print('') 
    print('Reading: '+f) 
    ds=Dataset(path+f,'r')
    if f=='Theta_bsoseI139_2013to2021_5dy.nc_2m':
      sose=ds.variables['Z'][:] 
      depth[:]=np.abs(sose)
      sose=ds.variables['THETA'][:,:,:,:]
      im=ma.getmaskarray(sose)
      im=sose==0.; sose[im]=0. 
      im=sose<=-5.; sose[im]=-5.
      # Define a 3D variable to hold the data
      thetao = ncfile.createVariable('thetao',np.float64,('time','depth','latitude','longitude')) # note: unlimited dimension is leftmost
      thetao.units = 'degrees_C' # degrees Kelvin
      thetao.long_name = 'Temperature' 
      print(thetao)
      #thetao[:,:,:,:]=sose
      thetao[:,:,0:li,:]=0.
      thetao[:,:,li:ll,:]=sose
      ds.close()
    elif f=='Salt_bsoseI139_2013to2021_5dy.nc_2m':
      sose=ds.variables['SALT'][:,:,:,:] 
      im=ma.getmaskarray(sose)
      im=sose==0.; sose[im]=33.9758
      im=sose>=40; sose[im]=40. 
      # Define a 3D variable to hold the data
      so = ncfile.createVariable('so',np.float64,('time','depth','latitude','longitude')) # note: unlimited dimension is leftmost
      so.units = '1e-3' # degrees Kelvin
      so.long_name = 'Salinity' 
      print(so)
      #so[:,:,:,:]=sose
      so[:,:,0:li,:]=33.9758
      so[:,:,li:ll,:]=sose
    elif f=='MLD_bsoseI139_2013to2021_5dy.nc':
      sose=ds.variables['BLGMLD'][:,:,:] 
      im=ma.getmaskarray(sose)
      sose[im]=22.1078 
      #im=sose==np.max(sose); sose[im]=22.1078 
      #im=sose==0.; sose[im]=22.1078 
      #im=sose>=50; sose[im]=50.
      #exit() 
      # Define a 3D variable to hold the data
      mlotst = ncfile.createVariable('mlotst',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
      mlotst.units = 'm' # degrees Kelvin
      mlotst.long_name = 'Density ocean mixed layer thickness' 
      print(mlotst)
      #mlotst[:,:,:]=sose
      mlotst[:,0:li,:]=22.1078
      mlotst[:,li:ll,:]=sose
    elif f=='SeaIceArea_bsoseI139_2013to2021_5dy.nc':
      sose=ds.variables['SIarea'][:,:,:] 
      im=ma.getmaskarray(sose)
      im=sose==0.; sose[im]=0. 
      # Define a 3D variable to hold the data
      siconc = ncfile.createVariable('siconc',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
      siconc.units = '1' # degrees Kelvin
      siconc.long_name = 'Ice concentration' 
      print(siconc)
      siconc[:,0:li,:]=0
      siconc[:,li:ll,:]=sose
    elif f=='SeaIceHeff_bsoseI139_2013to2021_5dy.nc':
      sose=ds.variables['SIheff'][:,:,:] 
      im=ma.getmaskarray(sose)
      im=sose==0.; sose[im]=0. 
      # Define a 3D variable to hold the data
      sithick = ncfile.createVariable('sithick',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
      sithick.units = 'm' 
      sithick.long_name = 'Sea ice thickness' 
      print(sithick)
      sithick[:,0:li,:]=0
      sithick[:,li:ll,:]=sose



ncfile.close(); exit()



