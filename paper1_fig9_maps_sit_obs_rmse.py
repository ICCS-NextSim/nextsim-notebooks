import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from netCDF4 import Dataset
from netCDF4 import date2num,num2date
import cmocean
import matplotlib 
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from matplotlib import dates
from mpl_toolkits.basemap import Basemap, addcyclic
#import cartopy
#import cartopy.crs as ccrs
import seapy
import irregular_grid_interpolator as myInterp
#from scipiy import interpolate
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utils import * 
import projection_info
from sys import exit
import os
import socket
import time as tictoc
import importlib
plt.ion()
plt.close('all')
importlib.reload(projection_info)
proj_info = projection_info.ProjectionInfo.sp_laea()
proj      = proj_info.pyproj

#Time
start_day  =24 # 6 vcorr serie initial day
start_month=7
start_year =2016
end_day    =25 #24 # bsie
end_month  =7  #8 sit
end_year   =2016

start_day  =1 # 6 vcorr serie initial day
start_month=1
start_year =2015
end_day    =31 # bsie 27/12/2021 = last day
end_month  =8  #8 sit
end_year   =2021


#Runs (names) or experiments (numbers - starts with 1)
exp=12
exptc=[12,31,exp] # if serie_or_map!=0
expt=exptc
expt=[12,31,19,30,18] # final expts (bsose, mevp, mevp+, bbm, bbm+)
expt=[31,30] # final expts (mevp, bbm)
#expt=[12,31,19,30] # final expts (bsose, mevp, mevp+, bbm)
#expt=[exp]

serie_or_maps=[0]#[1,2,3] # 1 for serie, 2 for video, and 3 for map, 0 for neither
my_dates=1
inc_obs=1
kmm=-1; # marker for seasonal maps 

#Variables
vname='sit_obs_rmse' # 'divergence'  # sit_obs_rmse_diff 
# sie, bsie,
# sit, siv, sit_rmse, (plot_maps) sit_obs_rmse, sit_obs_diff, sit_obs_rmse_diff
# siv, drift, vcorr, vcorr_diff, divergence, shear, processed variable e.g. 'bsie=(confusion matrix)', 'sit' 
# newice, newice_diff 'ridge_ratio' 'divergence' 

# Plot types
plot_scatter=0
plot_series =0
plot_hist   =0
plot_video  =0
plot_vchoice=0 # not working yet. it will for my webpage
plot_anim   =0 # solo video
plot_maps   =1 # seasonal maps
plot_mapo   =0 # maps with obs / based on plot_video and plot_smap
plot_smap   =0 # solo map

plot_cli    =0
save_fig    =1
plt_show    =1
interp_obs  =1 # only for SIE maps obs has 2x the model resolution
hist_norm   =0
####################################################################
# after BSOSE run (ocean boundary cond), m = mEVP, b = BBM
print(expt)
runs=['50km_ocean_wind'      ,'50km_bsose_20180102'   ,'50km_hSnowAlb_20180102','50km_61IceAlb_20180102','50km_14kPmax_20180102',       # 5
      '50km_20Clab_20180102' ,'50km_P14C20_20180102'  ,'50km_LandNeg2_20180102','50km_bsose_20130102'   ,'50km_dragWat01_20180102',     # 10
      '50km_glorys_20180102' ,'BSOSE'                 ,'50km_mevp_20130102'    ,'50km_lemieux_20130102' ,'50km_h50_20130102',           # 15
      '50km_hyle_20130102'   ,'50km_ckFFalse_20130102','50km_bWd020_20130102'  ,'mEVP+'                 ,'25km_bbm_20130102',           # 20
      '25km_mevp_20130102'   ,'12km_bbm_20130102'     ,'12km_mEVP_20130102'    ,'50km_bWd016_20130102'  ,'50km_mCd01_20130102',         # 25
      '50km_bCd01_20130102'  ,'50km_mWd016_20130102'  ,'50km_10kPcom_20130102' ,'50km_mevp10kP_20130102','BBM', # '50km_b10kP2h_20130102',       # 30
      'mEVP'                 ,'50km_b14kP1h_20130102' ,'50km_m14kP1h_20130102' ,'50km_b14kP2h_20130102' ,'50km_m14kP2h_20130102',       # 35
      '50km_mWd022_20130102' ,'50km_mWd024_20130102']       # ,'50km_mevp10kP_20130102']#  ,'50km_bCd01_20130102']         # 33

#Colors
if expt[0]==31:
  colors=['orange','b','pink','brown','g','r','k','yellow','orange','b','pink','brown','g','r','k','yellow']
elif expt[0]==12:
  colors=['pink','orange','b','black','brown','g','r','b','k','yellow','orange','b','pink','brown','g','r','k','yellow']
  colors=['r','orange','b','k','brown','g','r','b','k','yellow','orange','b','pink','brown','g','r','k','yellow']
else:
  colors=['k','orange','r','b','pink','brown','g','r','b','k','yellow','orange','b','pink','brown','g','r','k','yellow']

obs_colors=['g','y','orange'];

# varrays according to vname
if vname=='sic' or vname=='sie' or vname=='bsie':
  varray='sic' 
elif vname[0:3]=='sit' or vname=='siv': # or vname=='sit_rmse':
  varray='sit' 
elif vname=='drift' or vname[0:5]=='vcorr' or vname=='divergence' or vname=='shear':
  varray='siv' 
elif vname=='newice' or vname=='newice_diff':
  varray='newice' 
elif vname=='ridge_ratio':
  varray='ridge_ratio'

#trick to cover all months in runs longer than a year
end_month=end_month+1
ym_start= 12*start_year + start_month - 1
ym_end  = 12*end_year + end_month - 1
end_month=end_month-1

# series
params=[9,14];
# maps
paramm=[15];  
# profiles
parampr=[1, 2]; 
# time depth
paramtd=[9, 10, 11, 12]; 
# eddy stats
parame=[3, 4, 2, 5, 6, 8]; 

# SIE obs sources
obs_sources=['NSIDC','OSISAF-ease2'];
obs_sources=['OSISAF-ease2']#,'OSISAF-ease'] #['NSIDC','OSISAF','OSISAF-ease','OSISAF-ease2']: 

#paths
print('Hostname: '+socket.gethostname())
if socket.gethostname()[0:8]=='SC442555' or socket.gethostname()[0:10]=='wifi-staff':
  path_runs='/Users/rsan613/n/southern/runs/' # ''~/'
  path_fig ='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/figures/'
  path_data ='/Users/rsan613/n/southern/data/'
  path_bsose=path_data+'bsose/'
  #path_bsose='/Volumes/LaCie/mahuika/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/bsose/'
elif socket.gethostname()[0]=='w' or socket.gethostname()=='mahuika01' or socket.gethostname()=='mahuika':
  path_runs='/scale_wlg_persistent/filesets/project/uoa03669/rsan613/n/southern/runs/' # ''~/'
  #path_fig ='/scale_wlg_persistent/filesets/project/uoa03669/rsan613/n/southern/figures/' 
  path_fig='/scale_wlg_persistent/filesets/home/rsan613/figure/'
  path_data ='/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/'
  path_bsose='/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/bsose/'
else:
  print("Your data, runs and figures' paths haven't been set")
  exit()
  
#Grid information
run=runs[8]#expt[0]] # 'data_glorys'
data = xr.open_dataset(path_runs+run+'/output/Moorings_2018m01.nc')
lon_mod = data.longitude #sit.to_masked_array() # Extract a given variable
lat_mod = data.latitude #sit.to_masked_array() # Extract a given variable
#lon_mod=np.where(lon_mod!=np.max(lon_mod),lon_mod,179.99999999999)#180.01)
#lon_mod=np.where(lon_mod!=np.min(lon_mod),lon_mod,-179.99999999999)#-180.01)
lon_nex = lon_mod 
lat_nex = lat_mod 
v_spam=10
lon_modv=lon_mod[::v_spam,::v_spam]
lat_modv=lat_mod[::v_spam,::v_spam]
sit_output = data.sit.to_masked_array() # Extract a given variable
inan_mod=ma.getmaskarray(sit_output[0]); 
mask = ma.getmaskarray(sit_output[0]) #Get mask
lon_mod360=np.where(lon_mod>=0,lon_mod,lon_mod+360)

#ETOPO
filename=path_data+'etopo/ETOPO_Antarctic_10arcmin.nc'
print('Reading: '+filename)
ds=xr.open_dataset(filename)
lon_etopo=ds.variables['lon'][:]
lat_etopo=ds.variables['lat'][:]
h_etopo=ds.variables['z'][:]
ds.close()
lon_etopo,lat_etopo=np.meshgrid(lon_etopo,lat_etopo)
filename=path_data+'etopo/ETOPO_Antarctic_50km_nextsim.nc'
print('Loading: '+filename)
ds=xr.open_dataset(filename)
h_etopoi=ds.variables['__xarray_dataarray_variable__'][:]
ds.close()
filename=path_data+'etopo/ETOPO_Antarctic_drift.nc'
print('Loading: '+filename)
ds=xr.open_dataset(filename)
h_etopod=ds.variables['__xarray_dataarray_variable__'][:]
ds.close()

save_etopoi=0 
if save_etopoi==1: 
  print('Interpolating etopo bathy to nextsim grid')
  func=myInterp.IrregularGridInterpolator(np.array(lon_etopo),np.array(lat_etopo),np.array(lon_mod),np.array(lat_mod))
  h_etopoi=func.interp_field(np.array(h_etopo))
  h=xr.DataArray(h_etopoi) #,coords={'y': lat_e,'x': lon_e},dims=["y", "x"])
  filename=path_data+'etopo/ETOPO_Antarctic_50km_nextsim.nc'
  print('Saving: '+filename)
  h.to_netcdf(filename)
save_etopod=0
if save_etopod==1: 
  file=path_data+'/drift_osisaf_ease2/2016'+'/ice_drift_sh_ease2-750_cdr-v1p0_24h-20160101'+'1200.nc'; 
  print(file)
  data = xr.open_dataset(file)
  lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
  print('Interpolating etopo bathy to nextsim grid')
  func=myInterp.IrregularGridInterpolator(np.array(lon_etopo),np.array(lat_etopo),np.array(lon_obs),np.array(lat_obs))
  h_etopod=func.interp_field(np.array(h_etopo))
  h=xr.DataArray(h_etopod) #,coords={'y': lat_e,'x': lon_e},dims=["y", "x"])
  filename=path_data+'etopo/ETOPO_Antarctic_drift.nc'
  print('Saving: '+filename)
  h.to_netcdf(filename)
  exit()

## climatological time
#time_ini = dates.date2num(datetime.datetime(2015,1,1,3,0,0))
#time_fin = dates.date2num(datetime.datetime(2015,12,31,3,0,0)) 
#freqobs  = 1; # daily data
#times=pd.date_range(dates.num2date(time_ini), periods=int(time_fin-time_ini)*freqobs, freq=('%dD' % int(1/freqobs)))
#time_clin=dates.date2num(times)
#time_cli=dates.num2date(time_clin)
#time_clid=pd.DatetimeIndex(time_cli)


for serie_or_map in serie_or_maps:
  print(str(serie_or_map))

  # variables to be plotted vary with type (map or series)
  if serie_or_map==1: # series
    expt=exptc
    plot_series=1; plot_video=0; plot_maps=0;
    vnames=[ 'sie','bsie','sit','sit_rmse','siv','drift','vcorr'] # sie,bsie,sit,siv,drift,vcorr processed variable e.g. 'bsie=(confusion matrix)', 'sit' 
    varrays=['sic','sic' ,'sit','sit'     ,'sit','siv'  ,'siv'] # netcdf variable for each type of plot (raw variable used in xarray)
  elif serie_or_map==2: # video
    expt=[exp]
    plot_series=0; plot_video=1; plot_maps=0;
    vnames=[ 'sie','sit','drift'] 
    varrays=['sic','sit','siv'  ] 
  elif serie_or_map==3: # map
    expt=[exp]
    plot_series=0; plot_video=0; plot_maps=1;
    vnames=[ 'vcorr','sit'] 
    varrays=['siv'  ,'sit'] 
  elif serie_or_map==4: # video of choice
    expt=[exp]
    plot_series=0; plot_video=0; plot_maps=0; plot_vchoice=1;
    vnames=[ 'vcorr','sit'] 
    varrays=['siv'  ,'sit'] 
  else:
    vnames=[vname]; varrays=[varray]
  
  expts=range(len(runs)) #[0,1,2,3,4,5]
  expt=np.array(expt)-1

  # loop in all variables to be plotted
  for nvar in range(len(vnames)):
    vname=vnames[nvar]
    varray=varrays[nvar]
    # time will vary with type of variables
    if my_dates==0: 
      if varray=='sic':
        timedsdfsfdsf=1;
      elif varray=='sit':
        timedsdfsfdsf=1;
      elif varray=='siv':
        timedsdfsfdsf=1;
        # monthly sit
        #1/2013-8/2021
        #vcorr
        #1/2015-12/2021
  
    # time_obs
    time_ini = dates.date2num(datetime.datetime(start_year,start_month,start_day,3,0,0))
    time_fin = dates.date2num(datetime.datetime(end_year,end_month,end_day,3,0,0)) 
    freqobs  = 1; # daily data
    times=pd.date_range(dates.num2date(time_ini), periods=int(time_fin-time_ini)*freqobs, freq=('%dD' % int(1/freqobs)))
    time_obsn=dates.date2num(times)
    time_obs=dates.num2date(time_obsn)
    time_obsd=pd.DatetimeIndex(time_obs)
    time_obsni=[int(time_obsn[ii]) for ii in range(len(time_obsn))] # integer time for daily search
    time_obsni=np.array(time_obsni)
    timesix=pd.date_range(dates.num2date(time_ini), periods=int(time_fin-time_ini)*24/6, freq=('%dH' % int(6))) # time obs every 6h
    time_obsixn=dates.date2num(timesix)
    time_obsix=dates.num2date(time_obsixn)
    time_obsixd=pd.DatetimeIndex(time_obsix)
    
    # Loop in the experiments
    ke=0
    for ex in expt:
      ke+=1
      run=runs[expts[ex]]
    
      # Loading data
      if run=='BSOSE':
        if varray=='sic': #vname=='sie':
          filename=path_bsose+'SeaIceArea_bsoseI139_2013to2021_5dy.nc'
          print(filename)
          ds=xr.open_dataset(filename)
          sicc=ds.variables['SIarea'][:] 
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          vdatac=sicc
          #data = xr.open_dataset(filename)
          #timec = data.variables['time']; sicc = data.variables['SIarea']; 
        elif varray=='sit': #vname=='sie':
          filename=path_bsose+'SeaIceArea_bsoseI139_2013to2021_5dy.nc'
          print(filename)
          ds=xr.open_dataset(filename)
          sicc=ds.variables['SIarea'][:] 
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          filename=path_bsose+'SeaIceHeff_bsoseI139_2013to2021_5dy.nc'
          print(filename)
          ds=xr.open_dataset(filename)
          vdatac=ds.variables['SIheff'][:] 
          #sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          #vdatac=sicc
          #data = xr.open_dataset(filename)
          #timec = data.variables['time']; sicc = data.variables['SIarea']; 
        elif varray=='siv':
          filename=path_bsose+'SIuice_bsoseI139_2013to2021_5dy.nc'
          print(filename)
          ds=xr.open_dataset(filename)
          udatac=ds.variables['SIuice'][:] 
          filename=path_bsose+'SIvice_bsoseI139_2013to2021_5dy.nc'
          print(filename)
          ds=xr.open_dataset(filename)
          vdatac=ds.variables['SIvice'][:] 
    
        filename=path_bsose+'SeaIceArea_bsoseI139_2013to2021_5dy.nc'; ds=xr.open_dataset(filename)
        lon_sose=ds.variables['XC'][:]
        lon_sose=np.where(lon_sose<180,lon_sose,lon_sose-360)
        lat_sose=ds.variables['YC'][:]
        area_sose=ds.variables['rA'][:]/1000000.
        lon_mod,lat_mod=np.meshgrid(lon_sose,lat_sose);
        timec=ds.variables['time'][:]#/(3600*24) # making SOSE date centered as the 5-day average 
        ds.close()
        #time_date=dates.num2date(time_in,"seconds since 2012-12-01")
        #time_out=date2num(time_date,"hours since 1950-01-01 00:00:00")
        time_mod=dates.date2num(timec)
        time_mods=dates.num2date(time_mod)
        time_modd=pd.DatetimeIndex(time_mods)
        time_modi=[int(time_mod[ii]) for ii in range(len(time_mod))] # integer time for daily search
        time_modi=np.array(time_modi)
 
      else:
        k=0
        for ym in range( ym_start, ym_end ):
          k+=1
          y, m = divmod( ym, 12 ); m+=1
          filename=path_runs+run+'/output/Moorings_'+str(y)+'m'+str(m).zfill(2)+'.nc'
          print(filename)
          data = xr.open_dataset(filename)
          if k==1:
            #datac = data.variable[vname]
            timec = data.variables['time']; sicc = data.variables['sic']; vdatac = data.variables[varray]#['sit']
            if varray=='siv':
              udatac = data.variables['siu']
            #lon_mod = data.longitude #sit.to_masked_array() # Extract a given variable
            #lat_mod = data.latitude #sit.to_masked_array() # Extract a given variable
            v_spam=10
            lon_modv=lon_mod[::v_spam,::v_spam]
            lat_modv=lat_mod[::v_spam,::v_spam]
            sit_output = data.sit.to_masked_array() # Extract a given variable
            inan_mod=ma.getmaskarray(sit_output[0]); 
            mask = ma.getmaskarray(sit_output[0]) #Get mask
          else:
            #datac = xr.concat([datac,data],'time')
            time = data.variables['time'];   timec = xr.Variable.concat([timec,time],'time')
            sic = data.variables['sic'];   sicc = xr.Variable.concat([sicc,sic],'time')
            vdata = data.variables[varray]# ['sit']; 
            vdatac = xr.Variable.concat([vdatac,vdata],'time')
            if varray=='siv':
              udata = data.variables['siu']; 
              udatac = xr.Variable.concat([udatac,udata],'time')
            #exit() 
          data.close()
    
          lon_mod = lon_nex 
          lat_mod = lat_nex 
          time_mod=dates.date2num(timec)
          time_mods=dates.num2date(time_mod)
          time_modd=pd.DatetimeIndex(time_mods)
          time_modi=[int(time_mod[ii]) for ii in range(len(time_mod))] # integer time for daily search
          #exit()
        #datac.data_vars
      
      if plot_series==1:
        print('Ploting serie: '+vname+' '+run)
        plt.rcParams.update({'font.size': 22})
        # Plotting time series
        if ke==1:
          fig, ax = plt.subplots(1, 1, figsize = (16,8)) # landscape
          # plot obs
          if vname=='sie': # sea ice extent
            # loop in time to read obs
            kc=0; ll=[]
            for obs_source in obs_sources: 
              ll.append('Obs: '+obs_source); k=0; kc+=1
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:12]=='OSISAF-ease2':
                #if obs_source[0:11]=='OSISAF-ease':
                #  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                if obs_source[0:12]=='OSISAF-ease2':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease2-250_icdr-v2p0_20180101.nc';
                data = xr.open_dataset(file)
                xobs = data.variables['xc']; yobs = data.variables['yc']
                data.close()
                dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
              if obs_source=='NSIDC':
                file=path_data+'/sic_nsidc/2018'+'/'+'seaice_conc_daily_sh__20180101'+'_f17_v04r00.nc'
                data = xr.open_dataset(file)
                xobs = data.variables['xgrid']/1000.; yobs = data.variables['ygrid']/1000.
                data.close()
                dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
               
              for t in time_obs:
                k+=1
                if obs_source=='NSIDC':
                  file=path_data+'/sic_nsidc/'+t.strftime("%Y")+'/'+'seaice_conc_daily_sh__'+t.strftime("%Y%m%d")+'_f17_v04r00.nc'
                  print(file)
                  #obs_grid_area=25
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc']
                    #exit()
                  else:
                    sic_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'tdim' )
                elif obs_source[0:6]=='OSISAF':
                  #if obs_source[0:11]=='OSISAF-ease':
                  #  file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  if obs_source[0:12]=='OSISAF-ease2':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease2-250_icdr-v2p0_'+t.strftime("%Y%m%d")+'.nc'; 
                  else:
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_polstere-100_multi_'+t.strftime("%Y%m%d")+'.nc'
                    obs_grid_area=12.53377297 # 10 polstere
                  print(file)
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc']
                    #exit()
                  else:
                    sic_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'time' )
                data.close()
    
              print('Processing obs SIC to get extent')
              mean = np.zeros(np.shape(sicc_obs)[0])
              for t in range(np.shape(sicc_obs)[0]):
                print('Processing obs SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                #mean[t] = np.sum(sicc_obs[t]*25*25)
                sicct=sicc_obs[t]; 
                sicct=np.where(sicct<=1,sicct,np.nan); 
                siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
                #iext=np.where(sicct>1); sicct[iext]=0;
                #iext=np.where(sicct>.15)[0]; sicct[iext]=1;
                iext=np.where(sicct>.15); 
                for i in range(np.shape(iext)[1]):
                  siccz[iext[0][i],iext[1][i]]=1.
                #iext=np.where(sicct<=.15)[0]; sicct[iext]=0;
                #mean[t] = np.sum(sicct*25*25)
                if obs_source[0:11]=='OSISAF-ease' or obs_source[0:12]=='OSISAF-ease2' or obs_source=='NSIDC':
                  meant = np.multiply(siccz[0:-1,0:-1],obs_grid_area); # meant = np.multiply(meant,obs_grid_area);
                else:
                  meant = np.multiply(siccz,obs_grid_area); meant = np.multiply(meant,obs_grid_area);
                mean[t] = np.sum(meant)

              time=time_obs;
              mean=mean/1E6	
              if plot_cli==1:
                time,mean,std=daily_clim(time_obsd,mean)
                plt.plot(time, mean, color=obs_colors[kc-1])#,lw=2,alpha=0.5)   
                # plot randon points with colours for legend
                for exx in range(0,len(expt)):
                  plt.plot(time, mean, colors[exx])   
                plt.fill_between(time,mean-std,mean+std,facecolor=obs_colors[kc-1],alpha=0.5,lw=2)

              plt.plot(time, mean, color=obs_colors[kc-1])   
              plt.grid('on')
 
        if vname[0:3]=='sit':
          if inc_obs==0:
            if ke==1:
              ll=[]
            time=time_mods
            sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
            sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
            T = np.shape(sit)[0]
            mean = np.zeros(T)
            std = np.zeros(T)
            for t in range(T):
                mean[t] = np.mean((sit[t]*sic[t])/sic[t])
            plt.ylabel('SIT (m)'); plt.title('Domain average sea ice thickness (SIT)')
            ll.append(run+' mean = '+format(np.nanmean(mean),".2f"))
            figname='serie_sit_domain_average_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
          elif inc_obs==1:
            if ke==1: # if first expt, load obs and plot lines preping for legend
              kc=0; 
              # Loading data
              filename=path_data+'sit_cs2wfa/'+str(2015)+'/CS2WFA_25km_'+str(2015)+'0'+str(1)+'.nc'
              data = xr.open_dataset(filename); lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
              lon_obs=np.where(lon_obs<180,lon_obs,lon_obs-360)
              lon_obs=np.where(lon_obs!=np.max(lon_obs),lon_obs,180)
              lon_obs=np.where(lon_obs!=np.min(lon_obs),lon_obs,-180)
              sitc_obs = np.zeros([ym_end-ym_start,np.shape(sicc)[1],np.shape(sicc)[2]])
              timec=[]
              k=0; 
              for ym in range( ym_start, ym_end ):
                k+=1; y, m = divmod( ym, 12 ); m+=1
                filename=path_data+'sit_cs2wfa/'+str(y)+'/CS2WFA_25km_'+str(y)+str(m).zfill(2)+'.nc'
                print(filename)
                data = xr.open_dataset(filename,group='sea_ice_thickness')
                if k==1:
                  sitc = data.variables['sea_ice_thickness']; #vdatac = data.variables[varray]#['sit']
                  sitc=np.where(sitc>0,sitc, np.nan)
                  sitc=np.where(sitc<10,sitc, np.nan)
                  situ = data.variables['sea_ice_thickness_uncertainty']; #vdatac = data.variables[varray]#['sit']
                  situ=np.where(sitc>0,situ, np.nan); situ=np.where(sitc<10,situ, np.nan)
                else:
                  sit = data.variables['sea_ice_thickness']; sit=np.where(sit>0,sit, np.nan); sit=np.where(sit<10,sit, np.nan)
                  sitc = np.concatenate([sitc,sit],0) # 'time')
                  sit = data.variables['sea_ice_thickness_uncertainty']; sit=np.where(sit>0,sit, np.nan); sit=np.where(sit<10,sit, np.nan)
                  situ = np.concatenate([situ,sit],0) # 'time')
                timec.append(datetime.datetime(y,m,1))
                data.close()
              sicc_obs=sitc
              sicu_obs=situ

              if vname=='sit': 
                meau=np.nanmean(sicu_obs,axis=1); meau=np.nanmean(meau,axis=1)
                mean=np.nanmean(sicc_obs,axis=1); mean=np.nanmean(mean,axis=1)
                # for legend
                plt.plot(timec, mean,color=obs_colors[0])#,lw=2,alpha=0.5)   
                for exx in range(0,len(expt)):
                  plt.plot(timec, mean, colors[exx])   
                plt.ylim([0,2])

                plt.fill_between(timec,mean-meau,mean+meau,facecolor=obs_colors[kc],alpha=0.5,lw=2)
                #plt.plot(timec, mean, alpha=0.5)#color=obs_colors[kc],lw=2,alpha=0.5)   
                plt.plot(timec, mean,color=obs_colors[kc])#,lw=2,alpha=0.5)   
                plt.grid('on')
                ll=['CS2WFA mean = '+format(np.nanmean(sicc_obs),'.2f')]
              else:
                ll=[]
            
            vdatac=np.where(vdatac!=0,vdatac,np.nan)   
            sit_mod = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
            sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
            sicc_mod = np.zeros([ym_end-ym_start,np.shape(sicc_obs)[1],np.shape(sicc_obs)[2]])
            km=-1; time=[]
            st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
            func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
            et = tictoc.time()-st; print('Execution time:', et, 'seconds')
            for ym in range( ym_start, ym_end ):
              km+=1; y, m = divmod( ym, 12 ); m+=1
              print(run+': computing monthly mean for '+str(y)+'/'+str(m).zfill(2))
              iyear=time_modd.year==y
              imonth=time_modd.month==m; iym=np.where(iyear*imonth==True)
              time.append(time_mods[iym[0][0]])
              sit_modm=np.nanmean(sit_mod[iyear*imonth],axis=0) # month average
              sicc_mod[km]=func.interp_field(np.array(sit_modm))#[0]
              sicc_mod[km]=np.where(sicc_mod[km]>0,sicc_mod[km] , np.nan)
              sicc_mod[km]=np.where(sicc_mod[km]<10,sicc_mod[km] , np.nan)
              # masking mod where there is no obs
              #siccz=sicc_mod[km] 
              #cond=np.isnan(sicc_obs[km]); iext=np.where(cond==True)
              #for i in range(np.shape(iext)[1]):
              #  siccz[iext[0][i],iext[1][i]]=np.nan
              #sicc_mod[km]=siccz
    
            if vname=='sit': 
              # masking mod where there is no obs
              sicc_diff=sicc_obs+(sicc_mod-sicc_obs)
              #sicc_diff=sicc_mod
              mean=np.nanmean(sicc_diff,axis=1); mean=np.nanmean(mean,axis=1)
              print(run+' mean = '+format(np.nanmean(mean),".2f"))
              ll.append(run+' mean = '+format(np.nanmean(mean),".2f"))
              timec=time; 
              plt.ylabel('Sea ice thickness (m)'); plt.title('Monthly mean sea ice thickness')
            elif vname=='sit_rmse':
              # masking mod where there is no obs
              sicc_mod=sicc_obs+(sicc_mod-sicc_obs)
              mean=np.zeros_like(time)
              print('Computing monthly thickness rmse')
              for t in range(0,len(time)):
                mean[t]=np.sqrt(np.nanmean(np.square(np.subtract(sicc_obs[t],sicc_mod[t]))))
              ll.append(run+' mean = '+format(np.nanmean(mean),".2f"))
              timec=time; 
              plt.ylabel('Sea ice thickness rmse (m)'); plt.title('Sea ice thickness rmse [Model interp to Obs]')

            figname='serie_'+vname+'_month_mean_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
        elif vname=='sie':
          sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
          sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          diff=np.abs(int(time_obsni[0])-np.array(time_modi)); min_diff=np.min(diff)
          ifirst=np.where(diff==min_diff)[0][0]-1; 
          if ifirst<0:
            ifirst=0
          diff=np.abs(int(time_obsni[-1])-np.array(time_modi)); min_diff=np.min(diff)
          ilast=np.where(diff==min_diff)[-1][-1]+1
          #ilast=np.where(int(time_obsni[-1])==time_modi)[0][-1]
          sicc_mo=np.zeros((len(time_mod[ifirst:ilast+1]),np.shape(sic_mod)[1],np.shape(sic_mod)[2]))

          #exit()
          # GIVING BSOSE A DAILY DATASET
          if plot_cli==1 and run=='BSOSE':
            sicc_mo=np.zeros((len(time_obsni),np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
            iday2=-9999
            for t in range(len(time_obsni)): # (np.shape(sicc_mod)[0]):
              print('Concatenating BSOSE on day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              if iday!=iday2:
                sic_modi=sic[iday]
                iday2=iday;
              sicc_mo[t]=sic_modi
              sicc_mo[t]=np.where(sicc_mo[t]!=0.0,sicc_mo[t],np.nan)
            sic=sicc_mo;
            ifirst=0; ilast=len(time_obsni)
            time_mod=time_obsn;
            time_mods=time_obs;
            time_modd=time_obsd;

          # COMPUTING SIE
          T=len(time_mod[ifirst:ilast])
          mean = np.zeros(T);  std = np.zeros(T)
          k=-1
          for t in range(ifirst,ilast,1): # (np.shape(sicc_mod)[0]):
            k+=1
          #for t in range(T):
            print('Processing model SIC to get extent time: '+time_mods[t].strftime("%Y%m%d%HH:%MM"))
            #mean[t] = np.sum(sic[t]*50*50)
            sicct=sic[t];
            siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
            #iext=np.where(sicct>1)[0]; sicct[iext]=0;
            iext=np.where(sicct>.15)#[0]; sicct[iext]=1;
            for i in range(np.shape(iext)[1]):
              siccz[iext[0][i],iext[1][i]]=1.
            #iext=np.where(sicct<=.15)[0]; sicct[iext]=0;
            if run=='BSOSE':
              meant = np.multiply(siccz,area_sose)#; meant = np.multiply(meant,16);
            else:
              meant = np.multiply(siccz,25); meant = np.multiply(meant,25);
            mean[k] = np.sum(meant)

          time=timec[ifirst:ilast] 
          #plt.ylabel('Sea ice extent (km\^2)'); plt.title('Sea ice extent [sum(area[sic>.15])]')
          plt.ylabel('Sea ice extent ($10^6$ km'+'\u00B2'+')'); plt.title('Sea ice extent daily mean')
          figname='serie_sie_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
        elif vname=='bsie': # binary sea ice extent
          # loop in time to read obs
          if ke==1: # if first expt load obs
            kc=0; ll=[]
            for obs_source in obs_sources: 
              ll.append('OBS-'+obs_source); k=0; kc+=1
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:12]=='OSISAF-ease2':
                #if obs_source[0:11]=='OSISAF-ease':
                #  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                if obs_source[0:12]=='OSISAF-ease2':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease2-250_icdr-v2p0_20180101.nc';
                data = xr.open_dataset(file)
                lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
                xobs = data.variables['xc']; yobs = data.variables['yc']
                data.close()
                dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
                st = tictoc.time();   print('Creating weights to interp. obs to model grid ...'); # get the start time
                func=myInterp.IrregularGridInterpolator(np.array(lon_obs),np.array(lat_obs),np.array(lon_nex),np.array(lat_nex))#[0]
                et = tictoc.time()-st; print('Execution time:', et, 'seconds')
              for t in time_obs:
                k+=1
                if obs_source=='NSIDC':
                  file=path_data+'/sic_nsidc/'+t.strftime("%Y")+'/'+'seaice_conc_daily_sh__'+t.strftime("%Y%m%d")+'_f17_v04r00.nc'
                  print(file)
                  obs_grid_area=25
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc']
                    #exit()
                  else:
                    sic_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'tdim' )
                elif obs_source[0:6]=='OSISAF':
                  #if obs_source[0:11]=='OSISAF-ease':
                  #  file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  if obs_source[0:12]=='OSISAF-ease2':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease2-250_icdr-v2p0_'+t.strftime("%Y%m%d")+'.nc'; 
                  else:
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_polstere-100_multi_'+t.strftime("%Y%m%d")+'.nc'
                    obs_grid_area=12.53377297 # 10 polstere
                  print(file)
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc']
                  else:
                    sic_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'time' )
                data.close()
    
              print('Processing obs SIC to get extent')
              if interp_obs==1:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_nex)[0],np.shape(lon_nex)[1]])
              else:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_obs)[0],np.shape(lon_obs)[1]])
              for t in range(np.shape(sicc_obs)[0]):
                sicct=sicc_obs[t]; 
                if interp_obs==1:
                  sicobsi=func.interp_field(np.array(sicct))#[0]
                  # fixing gap due to interp method 
                  for tt in range(0,150): #226,np.shape(sicc_mod)[1]):  
                    sicobsi[tt][150]=sicobsi[tt][151] 
                  sicct=sicobsi
                siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
                iext=np.where(sicct>.15); 
                st = tictoc.time(); print('Processing obs SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM")) # get the start time
                for ii in range(np.shape(iext)[1]):
                  siccz[iext[0][ii],iext[1][ii]]=1.
                siccz[inan_mod]=np.nan
                sic_obs[t]=siccz
              sicc_obs=sic_obs
    
            # generating first plots for legend
            ll=[];
            for ki in range(len(expt)):
              plt.plot(np.nan,np.nan, colors[ki])   
    
          if run=='BSOSE':
            st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
            func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_nex),np.array(lat_nex))#[0]
            et = tictoc.time()-st; print('Execution time:', et, 'seconds')
      
          if ke>=1: # if first expt load obs
            sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
    
            diff=np.abs(int(time_obsni[0])-np.array(time_modi)); min_diff=np.min(diff)
            ifirst=np.where(diff==min_diff)[0][0]#-1; 
            if ifirst<0:
              ifirst=0
            diff=np.abs(int(time_obsni[-1])-np.array(time_modi)); min_diff=np.min(diff)
            ilast=np.where(diff==min_diff)[-1][-1]+1
            sicc_mo=np.zeros((len(time_mod[ifirst:ilast])+1,np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
            k=-1
            #Processing model SIC to get extent 
            for t in range(ifirst,ilast+1,1): # (np.shape(sicc_mod)[0]):
              k+=1
              print('Processing model SIC to get extent time: '+time_mods[t].strftime("%Y%m%d%HH:%MM"))
              sicct=sic_mod[t];
              sicc_mo[k]=sicct
    
            time_modi=time_modi[ifirst:ilast]
            # daily average
            if interp_obs==1:
              sicc_mod=np.zeros((len(time_obs),np.shape(lon_nex)[0],np.shape(lon_nex)[1]))
            else:
              sicc_mod=np.zeros((len(time_obs),np.shape(lon_obs)[0],np.shape(lon_obs)[1]))
            iday2=-9999
            for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
              if run=='BSOSE':
                # find the closest date
                diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
                iday=np.where(diff==min_diff)[0][0];
                print(iday)
                if iday!=iday2:
                  # interp to nextsim grid
                  #st = tictoc.time(); print('Interp BSOSE SIC to nextsim grid: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                  #siccz=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sicc_mo[iday]),np.array(lon_nex),np.array(lat_nex))[0]
                  siccz=func.interp_field(np.array(sicc_mo[iday])) 
                  #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
                  siccz[inan_mod]=np.nan
                  iday2=iday;
    
                print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                sicc_ex=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
                iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
                for ii in range(np.shape(iext)[1]):
                  sicc_ex[iext[0][ii],iext[1][ii]]=1.
                sicc_mod[t]=sicc_ex # np.nanmean(sicc_mo[iday,:,:],axis=0)
    
              else:  
                iday=np.where(time_obsni[t]==time_modi)[0]
                if interp_obs==1:
                  siccz=np.nanmean(sicc_mo[iday,:,:],axis=0)
                  print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                  sicct=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
                  iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
                  for ii in range(np.shape(iext)[1]):
                    sicct[iext[0][ii],iext[1][ii]]=1.
                  sicct[inan_mod]=np.nan
                  sicc_mod[t]=sicct
                else:
                  sicc_modm=np.nanmean(sicc_mo[iday,:,:],axis=0)
                  st = tictoc.time(); print('Interp model SIC to obs grid: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                  sicc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sicc_modm),np.array(lon_obs),np.array(lat_obs))[0]
                  et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
            mean=np.zeros([len(time_obs)]); mean[:]=np.nan
            mneg=mean.copy(); mpos=mean.copy()
            mtotal=mean.copy(); 
            sicc_diff=sicc_mod-sicc_obs; sicc_nan=sicc_mod
            izero=sicc_mod==0; sicc_nan[izero]=np.nan; sicc_nan=sicc_nan-sicc_obs;
            for t in range(len(time_obs)): 
              total    =np.sum(sicc_mod[t]==1); ontarget =np.sum(sicc_nan[t]==0)
              over     =np.sum(sicc_diff[t]==1); under    =np.sum(sicc_diff[t]==-1)
              mean[t]=100.*(ontarget/total); mneg[t]=100.*(under/total); mpos[t]=100.*(over/total)
              mtotal[t]=100.*((ontarget+over)/total); 
    

            time=time_obs

            if plot_cli==1:
              if run=='BSOSE':
                mneg=uniform_filter1d(mneg,10)
              time,mneg,std=daily_clim(time_obsd,mneg)
              #plt.plot(time, mean, obs_colors[ke-1])   
              # plot randon points with colours for legend
              #for exx in range(0,len(expt)):
              #  plt.plot(time, mean, colors[exx])   
              #plt.fill_between(time,mean-std,mean+std,facecolor=obs_colors[kc-1],alpha=0.5,lw=2)
#zzz

            #plt.plot(time, mtotal, colors[ke-1],linestyle=':')   
            plt.plot(time, mneg, colors[ke-1],linestyle='--')   
            #plt.plot(time, mpos, colors[ke-1],linestyle='-',marker='.')   


            plt.ylabel('(%)'); 
            #plt.title('True positive (-), false positive (.-) and false negative (--) model-obs comparison')
            #plt.title('Accurate (-), over- (.-) and underestimated (--) ice coverage')
            plt.title('Accurate (-) and underestimated (--) ice coverage')
            plt.grid('on')
            figname='serie_bsie_total_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
          
    
        elif vname=='siv':
          print('Plotting sea ice volume')
          if inc_obs==1:
            if ke==1: # if first expt load obs
              kc=0; 
              # Loading data
              filename=path_data+'sit_cs2wfa/'+str(2015)+'/CS2WFA_25km_'+str(2015)+'0'+str(1)+'.nc'

              data = xr.open_dataset(filename); lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
              lon_obs=np.where(lon_obs<180,lon_obs,lon_obs-360)
              lon_obs=np.where(lon_obs!=np.max(lon_obs),lon_obs,180)
              lon_obs=np.where(lon_obs!=np.min(lon_obs),lon_obs,-180)
              sitc_obs = np.zeros([ym_end-ym_start,np.shape(sicc)[1],np.shape(sicc)[2]])
              sicc_obs = np.zeros([ym_end-ym_start,np.shape(sicc)[1],np.shape(sicc)[2]])
              timec=[]
              k=0; 
              for ym in range( ym_start, ym_end ):
                k+=1; y, m = divmod( ym, 12 ); m+=1
                filename=path_data+'sit_cs2wfa/'+str(y)+'/CS2WFA_25km_'+str(y)+str(m).zfill(2)+'.nc'
                print(filename)
                data = xr.open_dataset(filename,group='sea_ice_thickness')
                data_sic = xr.open_dataset(filename)#,group='sea_ice_thickness')
                if k==1:
                  sitc = data.variables['sea_ice_thickness']; #vdatac = data.variables[varray]#['sit']
                  sitc=np.where(sitc>0,sitc, np.nan)
                  sitc=np.where(sitc<10,sitc, np.nan)
                  sico = data_sic.variables['sea_ice_concentration']; #vdatac = data.variables[varray]#['sit']
                  sico=np.where(sico>0,sico, np.nan)
                else:
                  sit = data.variables['sea_ice_thickness']; sit=np.where(sit>0,sit, np.nan); sit=np.where(sit<10,sit, np.nan)
                  sitc = np.concatenate([sitc,sit],0) # 'time')
                  sic = data_sic.variables['sea_ice_concentration']; sic=np.where(sic>0,sic, np.nan); 
                  sico = np.concatenate([sico,sic],0) # 'time')
                timec.append(datetime.datetime(y,m,28))
                data.close(); data_sic.close()
              sitc=(sitc/1000)*25*25
              sicc_obs=np.nansum(sitc,axis=1)
              mean=np.nansum(sicc_obs,axis=1); #mean=np.nanmean(mean,axis=1)
              plt.plot(timec, mean, color=obs_colors[kc],linestyle='--')   

              sitc=np.nanmean(sitc,axis=1); sitc=np.nanmean(sitc,axis=1)/1000; 
              sico=sico*25*25
              sico=np.nansum(sico,axis=1); sico=np.nansum(sico,axis=1); 
              #sicc_obs=np.nansum(sitc,axis=1)
              mean=sico*sitc
              plt.plot(timec, mean, color=obs_colors[kc+1],linestyle='--')   
              plt.grid('on')
              ll=['CS2WFA','CS2WA-mean']

          elif inc_obs==0 and ke==1:
            ll=[]

          if run=='BSOSE':
            sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
            siv = (sit/1000)*area_sose;  #_output = datac.sit.to_masked_array() # Extract a given variable
            sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          else:
            sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
            siv = sit*25*25/1000;  #_output = datac.sit.to_masked_array() # Extract a given variable
            sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          mean=np.sum(siv*sic,axis=1)
          mean=np.sum(mean,axis=1)
          #T = np.shape(sit)[0]
          #mean = np.zeros(T)
          time=time_mods
          #for t in range(T):
          #    mean[t] = np.sum((siv[t]*sic[t]))
          plt.xlim([time_obs[0],time_obs[-1]])
          plt.ylabel('SIV (km3)'); plt.title('Antarctic total sea ice volume (km3)')
          figname='serie_siv_total_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
        elif vname=='vcorr' or vname=='drift':
          if ke==1:
            k=0; kc=1;
            for t in time_obs:
              k+=1 # drift_osisaf_ease2
              file=path_data+'/drift_osisaf_ease2/'+t.strftime("%Y")+'/ice_drift_sh_ease2-750_cdr-v1p0_24h-'+t.strftime("%Y%m%d")+'1200.nc'; 
              print(file)
              data = xr.open_dataset(file)
              if k==1:
                u_obs = data.variables['dX'] #['cdr_seaice_conc']
                v_obs = data.variables['dY'] #['cdr_seaice_conc']
                uc_obs = data.variables['dX'] #['cdr_seaice_conc']
                vc_obs = data.variables['dY'] #['cdr_seaice_conc']
                lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
                v_spao=2
                lon_obsv=lon_obs[::v_spao,::v_spao]
                lat_obsv=lat_obs[::v_spao,::v_spao]
              else:
                u_obs = data.variables['dX'] #['cdr_seaice_conc']
                v_obs = data.variables['dY'] #['cdr_seaice_conc']
                uc_obs = xr.Variable.concat([uc_obs,u_obs] ,'time' )
                vc_obs = xr.Variable.concat([vc_obs,v_obs] ,'time' )
              data.close()
    
            uc_obs=np.array(uc_obs); vc_obs=np.array(vc_obs)
            if vname=='drift':
              magc_obs=np.sqrt(uc_obs**2+vc_obs**2)
              mean=np.nanmean(magc_obs,1); mean=np.nanmean(mean,1)
              time=time_obs

              if plot_cli==1:
                mean=uniform_filter1d(mean,10)
                time,mean,std=daily_clim(time_obsd,mean)
                plt.plot(time, mean, obs_colors[ke-1])   
                # plot randon points with colours for legend
                for exx in range(0,len(expt)):
                  plt.plot(time, mean, colors[exx])   
                plt.fill_between(time,mean-std,mean+std,facecolor=obs_colors[kc-1],alpha=0.5,lw=2)
#zzz
              plt.plot(time, mean, obs_colors[ke-1])   

            if vname=='drift':
              ll=['Obs: OSISAF-ease2']#+', mean='+format(np.nanmean(mean),".2f")]; 
              #ll=['Obs: OSISAF-ease2'+', mean='+format(np.nanmean(mean),".2f")]; 
              #ll=['OSI-455 mean = '+format(np.nanmean(mean),'.2f')]
            else:
              ll=[]; 
    
          st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
          func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
          et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          # model experiments
          u_mod = udatac*3.6*24;  v_mod = vdatac*3.6*24;
          uc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); uc_mod[:]=np.nan
          vc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); vc_mod[:]=np.nan
          mean = np.zeros([len(time_obs)]); mean[:]=np.nan
          # daily average
          iday2=-9999
          for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
            print('Computing model and obs vector complex correlation on day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
            if run=='BSOSE':
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              print(iday)
              if iday!=iday2:
                uc_modi=func.interp_field(np.array(u_mod[iday]))
                vc_modi=func.interp_field(np.array(v_mod[iday]))
                iday2=iday;
              uc_mod[t]=uc_modi
              vc_mod[t]=vc_modi
              uc_mod[t]=np.where(uc_mod[t]!=0.0,uc_mod[t],np.nan)
              vc_mod[t]=np.where(vc_mod[t]!=0.0,vc_mod[t],np.nan)
            else: # nextsim
              iday=np.where(time_obsni[t]==time_modi)[0]
              ucmod=np.nanmean(u_mod[iday,:,:],axis=0)
              vcmod=np.nanmean(v_mod[iday,:,:],axis=0)
              uc_mod[t]=func.interp_field(np.array(ucmod))
              vc_mod[t]=func.interp_field(np.array(vcmod))
              uc_mod[t]=np.where(uc_mod[t]!=0.0,uc_mod[t],np.nan)
              vc_mod[t]=np.where(vc_mod[t]!=0.0,vc_mod[t],np.nan)
    
            if vname=='vcorr':
              v_spave=3
              x=np.isfinite(vc_obs[t,::v_spave,::v_spave]+uc_mod[t,::v_spave,::v_spave])==1; 
              if np.sum(x==True)>1: 
                [vcorr,angle,X,Y]=veccor1(uc_obs[t,::v_spave,::v_spave],vc_obs[t,::v_spave,::v_spave],uc_mod[t,::v_spave,::v_spave],vc_mod[t,::v_spave,::v_spave])
                mean[t]=vcorr
            elif vname=='drift':
              uc_mod[t]=np.where(uc_obs[t]!=0,uc_mod[t],np.nan)
              vc_mod[t]=np.where(vc_obs[t]!=0,vc_mod[t],np.nan)
              magc_mod=np.sqrt(uc_mod[t]**2+vc_mod[t]**2)
              #magc_mod=np.where(magc_mod<=80.0,magc_mod,np.nan)
              mean[t]=np.nanmean(magc_mod)
          if vname=='vcorr':
            #ll.append(run+' - mean = '+format(np.nanmean(mean),".2f"))
            plt.ylabel('Complex correlation coef.'); plt.title('Complex correlation between modelled and obsverved sea ice drift')
            plt.ylim([0,1]) 
            figname='serie_vector_complex_correlation_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
          elif vname=='drift':
            #ll.append(run+' - mean = '+format(np.nanmean(mean),".2f"))
            plt.ylabel('Drift speed (km/day)'); plt.title('Antarctic sea-ice average drift speed (km/day)') 
            #plt.ylim([0,1]) 
            figname='serie_velocity_speed_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
          time=time_obs

        if plot_cli==1:
          if vname=='sie':
            if run=='BSOSE':
              mean=uniform_filter1d(mean,10)
            mean=mean/1E6	
            time,mean,std=daily_clim(time_modd,mean)
            ll.append(run)#+' - mean = '+format(np.nanmean(mean),".2f"))
            plt.ylim([0,25])
            plt.fill_between(time,mean-std,mean+std,facecolor=colors[ke-1],alpha=0.5,lw=2)

          elif vname=='bsie':
            if run=='BSOSE':
              mean=uniform_filter1d(mean,10)
            time,mean,std=daily_clim(time_obsd,mean)
            plt.ylim([0,100])
            ll.append(run)#+' - mean = '+format(np.nanmean(mean),".2f"))

          elif vname=='drift' or vname=='vcorr':
            #if run=='BSOSE':
            mean=uniform_filter1d(mean,10)
            #exit() 
            time,mean,std=daily_clim(time_obsd[0:len(mean)],mean)
            if vname=='drift':
              plt.ylim([2.5,20])
              ll.append(run)#+' - mean = '+format(np.nanmean(mean),".2f"))
            elif vname=='drift' or vname=='vcorr':
              plt.ylim([0,1])
              ll.append(run+' - mean = '+format(np.nanmean(mean),".2f"))
            #plt.fill_between(time,mean-std,mean+std,facecolor=colors[ke-1],alpha=0.5,lw=2)
#zzz
          else:
            time,mean,std=daily_clim(time_obsd,mean)
            ll.append(run+' - mean = '+format(np.nanmean(mean),".2f"))
          figname='cli_'+figname
          plt.plot(time, mean, colors[ke-1],linewidth=2)
        else:
          plt.plot(time, mean, colors[ke-1])   

        plt.grid('on')
        if ex==expt[-1]:
          #if vname!='sit':
          #  for i in expt:
          #    ll.append(runs[i]+' - mean = '+str(np.nanmean(mean)))
          plt.legend(ll)

          if plot_cli==1:
            date_form = dates.DateFormatter("%b")
          else:
            date_form = dates.DateFormatter("%b/%y")

          ax.xaxis.set_major_formatter(date_form)
          plt.tight_layout()
          if save_fig==1:
            if os.path.exists(path_fig+run)==False:
              os.mkdir(path_fig+run)

            figname=path_fig+run+'/'+figname
            print('Saving: '+figname)
            plt.savefig(figname)
          if plt_show==1:
            plt.show()

      if plot_hist==1:
        print('Ploting histogram: '+vname+' '+run)
        plt.rcParams.update({'font.size': 22})
        # Plotting time series
        # Ice divergence
        if vname[0:10]=='divergence': # if first expt load obs
          if ke==1:
            #fig=plt.figure(figsize = (16,8)) 
            fig, ax = plt.subplots(1, 3, figsize = (16,8)) # landscape
          time_obs=time_obsix
          u_mod = udatac*3.6*24;  v_mod = vdatac*3.6*24;
         
          # if BSOSE 
          uc_mod = np.zeros([len(time_obsix),np.shape(v_mod)[1],np.shape(v_mod)[2]]); uc_mod[:]=np.nan
          vc_mod = np.zeros([len(time_obsix),np.shape(v_mod)[1],np.shape(v_mod)[2]]); vc_mod[:]=np.nan
  
          # daily data (average if nextsim)
          iday2=-9999
          for t in range(len(time_obsix)): # (np.shape(sicc_mod)[0]):
            if run=='BSOSE':
              print('BSOSE day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              print(iday)
              if iday!=iday2:
                uc_modi=func.interp_field(np.array(u_mod[iday]))
                vc_modi=func.interp_field(np.array(v_mod[iday]))
                iday2=iday;
              uc_mod[t]=uc_modi
              vc_mod[t]=vc_modi
            else:
              #print('Cocatenating model drift results: '+time_obsix[t].strftime("%Y%m%d%HH:%MM"))
              iday=np.where(time_obsixn[t]==time_mod)[0]; #exit()
              iday=t # np.where(time_obsixn[t]==time_mod)[0]
              
              uc_mod[t]=u_mod[iday,:,:]
              vc_mod[t]=v_mod[iday,:,:]
              #uc_mod[t]=func.interp_field(np.array(ucmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
              #vc_mod[t]=func.interp_field(np.array(vcmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
    
          uc_mod=np.where(uc_mod!=0.0,uc_mod,np.nan)
          vc_mod=np.where(vc_mod!=0.0,vc_mod,np.nan)
          time=time_obsix; 

          dudx=(uc_mod[::,::,1::]-uc_mod[::,::,0:-1])/25.
          dvdx=(vc_mod[::,::,1::]-vc_mod[::,::,0:-1])/25.
          dudy=(uc_mod[::,1::,::]-uc_mod[::,0:-1,::])/25.
          dvdy=(vc_mod[::,1::,::]-vc_mod[::,0:-1,::])/25.
          div_mod=dudx[::,0:-1,::]+dvdy[::,::,0:-1]
          shear_mod=np.sqrt( np.square(dudx[::,0:-1,::]+dvdy[::,::,0:-1]) + np.square(dudy[::,0::,1::]+dvdx[::,1::,0::]) )

          hist_int=2E-2;
          model=div_mod; 
          div_mod=np.where(model>0.0,model,np.nan) # divergence
          con_mod=np.where(model<0.0,model,np.nan) # convergence

          hdiv=np.histogram(div_mod.flatten(),np.arange(0,np.nanmax(np.abs(model)),hist_int))
          hcon=np.histogram(con_mod.flatten(),np.arange(np.nanmin((con_mod)),0,hist_int))
          hshe=np.histogram(shear_mod.flatten(),np.arange(0,np.nanmax(shear_mod),hist_int))
          #hcon=np.histogram(con_mod.flatten(),np.arange(np.nanmin(con_mod),np.nanmax(con_mod),hist_int))
          #a=norm.pdf(div_mod.flatten(), loc=np.nanmean(div_mod.flatten()), scale=np.nanstd(div_mod.flatten()))
          
          if ke==1:
            ll=[]

          if hist_norm==1: 
            #ax[0].loglog(hdiv[1][0:-1],hdiv[0]/np.sum(hdiv[0][:]),#hdiv[0][0],
            #ax[0].loglog(hdiv[1][0:-1],hdiv[0]/hdiv[0][0],
            ax[0].loglog(hdiv[1][1::],hdiv[0]/np.max(hdiv[0][:]),
            color=colors[ke-1])
            #plt.ylim([0, 1])
          else:
            ax[0].loglog(hdiv[1][0:-1],hdiv[0],
            color=colors[ke-1])
          ax[0].set_title('Divergence')
          ax[0].set_xlabel('(d-1)')
          
          if ke==1:
            ll=[]
          if hist_norm==1: 
            ax[1].loglog(hcon[1][0:-1]*-1,hcon[0]/np.max(hcon[0][:]),
            color=colors[ke-1])
            #plt.ylim([0, 1])
          else:
            ax[1].loglog(hcon[1][0:-1]*-1,hcon[0],
            color=colors[ke-1])
          #plt.ylim([0, 1E2])
          ax[1].set_title('Convergence')
          ax[1].set_xlabel('(d-1)')

          #ax[2].bar(hshe[1][0:-1],hshe[0]/hshe[0][0],
          if hist_norm==1: 
            ax[2].loglog(hshe[1][1::],hshe[0]/np.max(hshe[0][:]),
            color=colors[ke-1])
            #plt.ylim([0, 1])
          else:
            ax[2].loglog(hshe[1][0:-1],hshe[0],
            color=colors[ke-1])
            
          #plt.ylim([0, 1E2])
          ax[2].set_title('Shear')
          ax[2].set_xlabel('(d-1)')

        if vname=='newice' or vname=='ridge_ratio': # if first expt load obs
          if ke==1:
            #fig=plt.figure(figsize = (16,8)) 
            fig, ax = plt.subplots(1, 3, figsize = (16,8)) # landscape
          time_obs=time_obsix
          model=vdatac
          if vname=='newice':# or vname=='ridge_ratio': # if first expt load obs
            hist_int=2E-3; #E-2
          elif vname=='ridge_ratio': # if first expt load obs
            #model=model*100.
            hist_int=2E-3;
          model=np.where(model!=0.0,model,np.nan) # convergence

          hist_int=2E-3;
          hdiv=np.histogram(model.flatten(),np.arange(0,np.nanmax(np.abs(model)),hist_int))
          #hcon=np.histogram(con_mod.flatten(),np.arange(np.nanmin(con_mod),np.nanmax(con_mod),hist_int))
          #a=norm.pdf(div_mod.flatten(), loc=np.nanmean(div_mod.flatten()), scale=np.nanstd(div_mod.flatten()))
          
          if ke==1:
            #ax1=fig.add_subplot(1,3,1)
            ll=[]
          #ax[0].bar(hdiv[1][0:-1],hdiv[0]/hdiv[0][0],

          if vname=='ridge_ratio': # if first expt load obs
            ax[0].semilogy(hdiv[1][0:-1],hdiv[0],#/hdiv[0][0],
            color=colors[ke-1])
          else:
            ax[0].loglog(hdiv[1][0:-1],hdiv[0],#/hdiv[0][0],
            color=colors[ke-1])
          ax[0].grid('on')
          #plt.ylim([0, 1E2])
          if vname=='newice':# or vname=='ridge_ratio': # if first expt load obs
            ax[0].set_title('Ice growth')
            ax[0].set_xlabel('(m/day)')
          elif vname=='ridge_ratio': # if first expt load obs
            ax[0].set_title('Ridging ratio')
            ax[0].set_xlabel('(Ridged ice) / (Total ice)')

        if ex==expt[-1]:
          for i in expt:
            ll.append(runs[i])
    
          #ax1=fig.add_subplot(1,3,1)
          ax[0].legend(ll)
          plt.tight_layout()

        figname=path_fig+run+'/histogram_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
        fig.tight_layout() 
        if ex==expt[-1]:
          if save_fig==1:
            if os.path.exists(path_fig+run)==False:
              os.mkdir(path_fig+run)
            print('Saving: '+figname)
            plt.savefig(figname,dpi=300,bbox_inches='tight')
          if plt_show==1:
            plt.show()
      
      ### Plot maps (seasonal) 
      if plot_maps==1:
        print('Ploting map: '+vname+' '+run)
        plt.rcParams.update({'font.size': 12})
        if ex==expt[0]:
          fig=plt.figure(figsize = (9,8)) # square
        if vname[0:5]=='vcorr' or vname=='drift': 
          if ke==1 : # if first expt load obs
            ll=[]; k=0
            for t in time_obs:
              k+=1 # drift_osisaf_ease2
              file=path_data+'/drift_osisaf_ease2/'+t.strftime("%Y")+'/ice_drift_sh_ease2-750_cdr-v1p0_24h-'+t.strftime("%Y%m%d")+'1200.nc'; 
              print(file)
              data = xr.open_dataset(file)
              if k==1:
                u_obs = data.variables['dX'] #['cdr_seaice_conc']
                v_obs = data.variables['dY'] #['cdr_seaice_conc']
                uc_obs = data.variables['dX'] #['cdr_seaice_conc']
                vc_obs = data.variables['dY'] #['cdr_seaice_conc']
                lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
                v_spao=2
                lon_obsv=lon_obs[::v_spao,::v_spao]
                lat_obsv=lat_obs[::v_spao,::v_spao]
              else:
                u_obs = data.variables['dX'] #['cdr_seaice_conc']
                v_obs = data.variables['dY'] #['cdr_seaice_conc']
                uc_obs = xr.Variable.concat([uc_obs,u_obs] ,'time' )
                vc_obs = xr.Variable.concat([vc_obs,v_obs] ,'time' )
              data.close()
            uc_obs=np.array(uc_obs); vc_obs=np.array(vc_obs)
            st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
            func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
            et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          # model experiments
          u_mod = udatac*3.6*24;  v_mod = vdatac*3.6*24;
          uc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); uc_mod[:]=np.nan
          vc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); vc_mod[:]=np.nan
          mean = np.zeros([len(time_obs)]); mean[:]=np.nan
          # daily data (average if nextsim)
          iday2=-9999
          for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
            if run=='BSOSE':
              print('BSOSE day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              print(iday)
              if iday!=iday2:
                uc_modi=func.interp_field(np.array(u_mod[iday]))
                vc_modi=func.interp_field(np.array(v_mod[iday]))
                iday2=iday;
              uc_mod[t]=uc_modi
              vc_mod[t]=vc_modi
              uc_mod[t]=np.where(uc_mod[t]!=0.0,uc_mod[t],np.nan)
              vc_mod[t]=np.where(vc_mod[t]!=0.0,vc_mod[t],np.nan)
            else:
              print('Interping model to obs lonlat on day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              iday=np.where(time_obsni[t]==time_modi)[0]
              ucmod=np.nanmean(u_mod[iday,:,:],axis=0)
              vcmod=np.nanmean(v_mod[iday,:,:],axis=0)
              uc_mod[t]=func.interp_field(np.array(ucmod))
              vc_mod[t]=func.interp_field(np.array(vcmod))
              uc_mod[t]=np.where(uc_mod[t]!=0.0,uc_mod[t],np.nan)
              vc_mod[t]=np.where(vc_mod[t]!=0.0,vc_mod[t],np.nan)
          # loop in the four seasons
          km=-1; tseason=['JFM','ASO','JAS','OND']
          for m in [1,4]:
            km+=1; 
            print(run+': computing seasonal complex vector correlation starting in month '+str(m).zfill(2))
            if m==1:
              imonth1=time_obsd.month==1; imonth2=time_obsd.month==2; imonth3=time_obsd.month==3; 
            if m==4: 
              imonth1=time_obsd.month==8; imonth2=time_obsd.month==9; imonth3=time_obsd.month==10; 
            if m==7: 
              imonth1=time_obsd.month==7; imonth2=time_obsd.month==8; imonth3=time_obsd.month==9; 
            if m==10: 
              imonth1=time_obsd.month==10; imonth2=time_obsd.month==11; imonth3=time_obsd.month==12; 
            umod=np.concatenate((uc_mod[imonth1],uc_mod[imonth2],uc_mod[imonth3]),0)
            vmod=np.concatenate((vc_mod[imonth1],vc_mod[imonth2],vc_mod[imonth3]),0)
            uobs=np.concatenate((uc_obs[imonth1],uc_obs[imonth2],uc_obs[imonth3]),0)
            vobs=np.concatenate((vc_obs[imonth1],vc_obs[imonth2],vc_obs[imonth3]),0)
    
            if vname=='drift': 
              umean=umod[:,::v_spao,::v_spao]
              vmean=vmod[:,::v_spao,::v_spao]
              umean = np.nanmean(umean,axis=0) 
              vmean = np.nanmean(vmean,axis=0) 
              mean = np.sqrt(umod**2+vmod**2)
              mean = np.nanmean(mean,axis=0) 
            elif vname[0:5]=='vcorr': 
              # loop for each grid point
              mean = np.zeros([np.shape(uobs)[1],np.shape(uobs)[2]]); mean[:]=np.nan
              for i in range(np.shape(uobs)[1]):
                for ii in range(np.shape(uobs)[2]):
                 uc_mo=umod[::,i,ii]; vc_mo=vmod[::,i,ii]
                 uc_ob=uobs[::,i,ii]; vc_ob=vobs[::,i,ii]
                 x=np.isfinite(uc_ob+uc_mo)==1; 
                 if np.sum(x==True)>1: 
                 #if np.sum(np.isnan(uc_ob))+1<len(uc_ob) and np.sum(np.isnan(uc_mo))+1<len(uc_mo): 
                   uc_ob=uc_ob[x]; vc_ob=vc_ob[x]; uc_mo=uc_mo[x]; vc_mo=vc_mo[x]
                   vcorr,angle,X,Y=veccor1(uc_ob,vc_ob,uc_mo,vc_mo) 
                   mean[i][ii]=vcorr

            # computing difference between 2 expts
            if vname=='vcorr_diff':
              if ex==expt[-1]:
                print('Last experiment')
                mean=(means[km,:,:]-mean)#*90
              else:
                print('First experiment')
                if km==0:
                  means=np.zeros((4,np.shape(mean)[0],np.shape(mean)[1]))
                means[km,:,:]=mean

            if ex==expt[-1]:
              ax=fig.add_subplot(2,2,km+1)
              bm = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l')#,ax=ax[km])
              bm.drawcoastlines()
              bm.fillcontinents(color='grey',lake_color='aqua')
              # add wrap-around point in longitude.
              #mean,lon_obs = addcyclic (mean,lon_obs)
              longr, latgr = bm([0,0],[-90,-70.5])#,inverse=True)
              bm.plot(longr,latgr,color='grey',linewidth=2)
              lonp, latp = bm(lon_obs,lat_obs)#,inverse=True)
              lonov, latov = bm(lon_obsv,lat_obsv)#,inverse=True)
              if vname=='vcorr_diff':
                plt.title(tseason[km]+' '+runs[expt[0]]+' corr. - '+runs[expt[1]]+' corr.',loc='center')
                cmap = cmocean.cm.balance
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=-1,vmax=1)
                # contour
                ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
                clevels=[0]# np.linspace(0,40,40,endpoint=False)
                ic=bm.contour(lonp,latp,mean,clevels,colors=('k'),linewidths=(1.5,),origin='upper',linestyles='solid',extent=ext)
              elif vname=='vcorr':
                plt.title(tseason[km]+' '+run+' vel. corr.',loc='center')
                cmap = cmocean.cm.amp
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=0,vmax=1)
                # contour
                ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
                clevels=[.7,1.]# np.linspace(0,40,40,endpoint=False)
                ic=bm.contour(lonp,latp,mean,clevels,colors=('k'),linewidths=(1.5,),origin='upper',linestyles='solid',extent=ext)
                #ic.clabel(clevels,fmt='%2.1f',colors='w',fontsize=20)
              elif vname=='drift':
                plt.title(tseason[km]+' '+run+' vel. corr.',loc='center')
                cmap = cmocean.cm.tempo
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=0,vmax=40)
                # contour
                #ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
                #clevels=[.7,1.]# np.linspace(0,40,40,endpoint=False)
                #ic=bm.contour(lonp,latp,mean,clevels,colors=('k'),linewidths=(1.5,),origin='upper',linestyles='solid',extent=ext)
                #ic.clabel(clevels,fmt='%2.1f',colors='w',fontsize=20)
                im11 = bm.quiver(lonov, latov, umean, vmean,color='black',width=0.002,scale=500.0)
                qk=plt.quiverkey(im11,.1,.1,10,'10 km/day',labelpos='S',fontproperties={'size':8})


              # computing stats per subregion
              text_fig=0
              if text_fig==1:
                lon_regions=[-150,-61,-20,34,90,160];
                text_map_w_stats(mean,lon_obs,bm,lon_regions,'mean','','black')
                plt.annotate('Total mean: '+format(np.nanmean(mean),'.2f')+'', xy=(.3,.56), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
                dataf=np.where(h_etopod>=-800,mean,np.nan); dataf=format(np.nanmean(dataf),'.2f')
                plt.annotate('Coastal mean: '+dataf+'', xy=(.3,.51), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
                dataf=np.where(h_etopod<-800,mean,np.nan); dataf=format(np.nanmean(dataf),'.2f')
                plt.annotate('Deep mean: '+dataf+'', xy=(.3,.46), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
              # including colorbar
              divider = make_axes_locatable(ax)
              cax = divider.append_axes('right', size='5%', pad=0.05)
              fig.colorbar(im1, cax=cax, orientation='vertical')

          figname=path_fig+run+'/map_vector_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'

        elif vname[0:3]=='sie' or vname[0:3]=='sic': # if first expt load obs
          # plot obs
          if ke==1:# and vname[0:7]=='sie_obs': # _diff' or _rmse
            # loop in time to read obs
            kc=0; ll=[]
            for obs_source in obs_sources: 
              ll.append('OBS-'+obs_source); k=0; kc+=1
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:12]=='OSISAF-ease2':
                #if obs_source[0:11]=='OSISAF-ease':
                #  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                if obs_source[0:12]=='OSISAF-ease2':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease2-250_icdr-v2p0_20180101.nc';
                data = xr.open_dataset(file)
                lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
                xobs = data.variables['xc']; yobs = data.variables['yc']
                data.close()
                dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
                st = tictoc.time();   print('Creating weights to interp. obs to model grid ...'); # get the start time
                func=myInterp.IrregularGridInterpolator(np.array(lon_obs),np.array(lat_obs),np.array(lon_nex),np.array(lat_nex))#[0]
                et = tictoc.time()-st; print('Execution time:', et, 'seconds')
              for t in time_obs:
                k+=1
                if obs_source=='NSIDC':
                  file=path_data+'/sic_nsidc/'+t.strftime("%Y")+'/'+'seaice_conc_daily_sh__'+t.strftime("%Y%m%d")+'_f17_v04r00.nc'
                  print(file)
                  obs_grid_area=25
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc']
                    #exit()
                  else:
                    sic_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'tdim' )
                elif obs_source[0:6]=='OSISAF':
                  #if obs_source[0:11]=='OSISAF-ease':
                  #  file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  if obs_source[0:12]=='OSISAF-ease2':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease2-250_icdr-v2p0_'+t.strftime("%Y%m%d")+'.nc'; 
                  else:
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_polstere-100_multi_'+t.strftime("%Y%m%d")+'.nc'
                    obs_grid_area=12.53377297 # 10 polstere
                  print(file)
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc']
                  else:
                    sic_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'time' )
                data.close()
    
              print('Processing obs SIC to get extent')
              if interp_obs==1:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_nex)[0],np.shape(lon_nex)[1]])
              else:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_obs)[0],np.shape(lon_obs)[1]])
              for t in range(np.shape(sicc_obs)[0]):
                sicct=sicc_obs[t]; 
                if interp_obs==1:
                  st = tictoc.time();   print('Interping obs to model grid ...'); # get the start time
                  sicobsi=func.interp_field(np.array(sicct))#[0]
                  # fixing gap due to interp method 
                  for tt in range(0,150): #226,np.shape(sicc_mod)[1]):  
                    sicobsi[tt][150]=sicobsi[tt][151] 
                  et = tictoc.time()-st; print('Execution time:', et, 'seconds')
                  sicct=sicobsi

                #if vname=='sie':
                #  siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
                #  iext=np.where(sicct>.15); 
                #  st = tictoc.time(); print('Processing obs SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM")) # get the start time
                #  for ii in range(np.shape(iext)[1]):
                #    siccz[iext[0][ii],iext[1][ii]]=1.
                #  siccz[inan_mod]=np.nan
                #  sic_obs[t]=siccz
                #else:
                sicct[inan_mod]=np.nan
                sic_obs[t]=sicct
    
              sicc_obs=sic_obs

          if ke>=1: # if first expt load obs
            sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
            sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
    
            diff=np.abs(int(time_obsni[0])-np.array(time_modi)); min_diff=np.min(diff)
            ifirst=np.where(diff==min_diff)[0][0]#-1; 
            if ifirst<0:
              ifirst=0
            diff=np.abs(int(time_obsni[-1])-np.array(time_modi)); min_diff=np.min(diff)
            ilast=np.where(diff==min_diff)[-1][-1]+1
            sicc_mo=np.zeros((len(time_mod[ifirst:ilast])+1,np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
            k=-1
            #Processing model SIC to get extent 
            for t in range(ifirst,ilast+1,1): # (np.shape(sicc_mod)[0]):
              k+=1
              print('Processing model SIC to get extent time: '+time_mods[t].strftime("%Y%m%d%HH:%MM"))
              sicct=sic_mod[t];
              sicc_mo[k]=sicct
  
            if run=='BSOSE':
              st = tictoc.time();   print('Creating weights to interp. BSOSE to model grid ...'); # get the start time
              func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_nex),np.array(lat_nex))#[0]
              et = tictoc.time()-st; print('Execution time:', et, 'seconds')
  
            time_modi=time_modi[ifirst:ilast]
            # daily average
            if interp_obs==1:
              sicc_mod=np.zeros((len(time_obs),np.shape(lon_nex)[0],np.shape(lon_nex)[1]))
            else:
              sicc_mod=np.zeros((len(time_obs),np.shape(lon_obs)[0],np.shape(lon_obs)[1]))
            iday2=-9999
            for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
              if run=='BSOSE':
                # find the closest date
                diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
                iday=np.where(diff==min_diff)[0][0];
                print(iday)
                if iday!=iday2:
                  # interp to nextsim grid
                  siccz=func.interp_field(np.array(sicc_mo[iday]))#[0]
                  siccz[inan_mod]=np.nan
                  iday2=iday;
    
                print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                sicc_ex=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
                iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
                for ii in range(np.shape(iext)[1]):
                  sicc_ex[iext[0][ii],iext[1][ii]]=1.
                sicc_mod[t]=sicc_ex # np.nanmean(sicc_mo[iday,:,:],axis=0)
    
              else:  
                iday=np.where(time_obsni[t]==time_modi)[0]
                if interp_obs==1:
                  siccz=np.nanmean(sicc_mo[iday,:,:],axis=0)
                  #exit()
                  #sicc_mod[t]=siccz # np.nanmean(sicc_mo[iday,:,:],axis=0)
                  #if vname=='sie':
                  #  print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                  #  sicct=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
                  #  iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
                  #  for ii in range(np.shape(iext)[1]):
                  #    sicct[iext[0][ii],iext[1][ii]]=1.
                  #  sicct[inan_mod]=np.nan
                  #  sicc_mod[t]=sicct
                  #else:
                  siccz[inan_mod]=np.nan
                  sicc_mod[t]=siccz
                else:
                  sicc_modm=np.nanmean(sicc_mo[iday,:,:],axis=0)
                  st = tictoc.time(); print('Interp model SIC to obs grid: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                  sicc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sicc_modm),np.array(lon_obs),np.array(lat_obs))[0]
                  et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    

          #sicc_diff=sicc_mod-sicc_obs 

          # loop in the four seasons
          km=-1; tseason=['JFM','ASO','JAS','OND']
          for m in [1,4]:
            km+=1; 
            kmm+=1; 
            print(run+': computing seasonal complex vector correlation starting in month '+str(m).zfill(2))
            if m==1:
              imonth1=time_obsd.month==1; imonth2=time_obsd.month==2; imonth3=time_obsd.month==3; 
            if m==4: 
              imonth1=time_obsd.month==8; imonth2=time_obsd.month==9; imonth3=time_obsd.month==10; 
            if m==7: 
              imonth1=time_obsd.month==7; imonth2=time_obsd.month==8; imonth3=time_obsd.month==9; 
            if m==10: 
              imonth1=time_obsd.month==10; imonth2=time_obsd.month==11; imonth3=time_obsd.month==12; 
            sic_mods=np.concatenate((sicc_mod[imonth1],sicc_mod[imonth2],sicc_mod[imonth3]),0)
            sic_obss=np.concatenate((sicc_obs[imonth1],sicc_obs[imonth2],sicc_obs[imonth3]),0)

            mmod=np.nanmean(sic_mods,0)
            mobs=np.nanmean(sic_obss,0)
            if vname=='sie':
              siccz=np.zeros((np.shape(mmod)[0],np.shape(mmod)[1])) 
              sicco=np.zeros((np.shape(mmod)[0],np.shape(mmod)[1])) 
              iext=np.where(mmod>.15); 
              ioxt=np.where(mobs>.15); 
              for ii in range(np.shape(iext)[1]):
                siccz[iext[0][ii],iext[1][ii]]=1.
              siccz[inan_mod]=np.nan
              mmod=siccz
              for ii in range(np.shape(ioxt)[1]):
                sicco[ioxt[0][ii],ioxt[1][ii]]=1.
              sicco[inan_mod]=np.nan
              mobs=sicco

            mean=mmod-mobs
            mean=np.where(mean!=0,mean,np.nan)
#zzz
            #exit()
            if ex==ex: # pt[-1]:
              ax=fig.add_subplot(2,2,kmm+1)
              bm = Basemap(projection='splaea',boundinglat=-52,lon_0=180,resolution='l')#,ax=ax[km])
              bm.drawcoastlines(linewidth=.5)
              bm.fillcontinents(color='grey',lake_color='aqua')
              # add wrap-around point in longitude.
              #mean,lon_obs = addcyclic (mean,lon_obs)
              longr, latgr = bm([0,0],[-90,-70.5])#,inverse=True)
              bm.plot(longr,latgr,color='grey',linewidth=2)
              #bm.drawparallels(np.arange(-90,-30,5))
              #bm.drawmeridians(np.arange(0,360,30))
              lonp, latp = bm(lon_mod,lat_mod)#,inverse=True)
              ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
              if vname=='sie' or vname=='sic':
                plt.title(tseason[km]+' '+runs[ex]+' - Obs.',loc='center')
                cmap = cmocean.cm.balance
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=-2.,vmax=2.)
                #ic=bm.contour(lonp,latp,mmod,[1.],colors=('magenta'),linewidths=(.5,),origin='upper',linestyles='solid',extent=ext)
                ic=bm.contour(lonp,latp,mobs,[1.],colors=('green'),linewidths=(1.,),origin='upper',linestyles='solid',extent=ext)
              # contour
              lone, late = bm(lon_etopo,lat_etopo)#,inverse=True)
              ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
              clevels=[-800] # np.linspace(0,40,40,endpoint=False)
              #ic=bm.contour(lonp,latp,h_etopoi,clevels,colors=('magenta'),linewidths=(.5,),origin='upper',linestyles='solid',extent=ext)
              #ic.clabel(clevels,fmt='%2.1f',colors='w',fontsize=20)
              # computing stats per subregion
              text_fig=1
              if text_fig==1:
                mean=np.where(mean<0,-1.*mean,mean); #mneg=np.where(mean<=0,mean,np.nan); 
                mpos=np.where(mean>=0,mean,np.nan); mneg=np.where(mean<=0,mean,np.nan); 
                mean=(mean*25.*25.)/1e6; mobs=(mobs*25.*25.)/1e6; mmod=(mmod*25.*25.)/1e6
                mpos=(mpos*25.*25.)/1e6; mneg=(mneg*25.*25.)/1e6; #mmod=(mmod*25.*25.)/1e6
                lon_regions=[-150,-61,-20,34,90,160]; lat_regions=[ -77,-75,-73,-68.5,-67,-70];
                text_map_w_stats(mean,lon_mod,bm,lon_regions,lat_regions,'sum','M $\mathrm{km^{2}}$','black')
                plt.annotate('Total obs.: '+format(np.nansum(mobs),'.2f')+r' M $\mathrm{km^{2}}$', xy=(.3,.56), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
                plt.annotate('Total mod.: '+format(np.nansum(mmod),'.2f')+r' M $\mathrm{km^{2}}$', xy=(.3,.51), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
                dataf=np.where(h_etopoi>=-800,mean,np.nan); dataf=format(np.nanmean(dataf),'.2f')
                #plt.annotate('Coastal mean: '+dataf+'', xy=(.3,.51), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
                dataf=np.where(h_etopoi<-800,mean,np.nan); dataf=format(np.nanmean(dataf),'.2f')
                #plt.annotate('Deep mean: '+dataf+'', xy=(.3,.46), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',

              # including colorbar
              #divider = make_axes_locatable(ax)
              #cax = divider.append_axes('right', size='5%', pad=0.05)
              #fig.colorbar(im1, cax=cax, orientation='vertical')

          figname=path_fig+run+'/map_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'



        elif vname[0:3]=='sit':  
          #timec=time_obs
          #time_ob=dates.date2num(timec); time_obss=dates.num2date(time_ob); time_obsd=pd.DatetimeIndex(time_obss)
          if vname=='sit' or vname[0:7]=='sit_obs': # _diff' or _rmse
            if ke==1: # if first expt load obs
              kc=0; 
              # Loading data
              filename=path_data+'sit_cs2wfa/'+str(2015)+'/CS2WFA_25km_'+str(2015)+'0'+str(1)+'.nc'

              data = xr.open_dataset(filename); lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
              lon_obs=np.where(lon_obs<180,lon_obs,lon_obs-360); lon_obs=np.where(lon_obs!=np.max(lon_obs),lon_obs,180)
              lon_obs=np.where(lon_obs!=np.min(lon_obs),lon_obs,-180); sitc_obs = np.zeros([ym_end-ym_start,np.shape(sicc)[1],np.shape(sicc)[2]])
              timec=[]; ll=['CS2WFA']; k=0; 
              print('Creating weights to interp. model to obs grid ...'); # get the start time
              func=myInterp.IrregularGridInterpolator(np.array(lon_obs),np.array(lat_obs),np.array(lon_mod),np.array(lat_mod))#[0]
              for ym in range( ym_start, ym_end ):
                k+=1; y, m = divmod( ym, 12 ); m+=1
                filename=path_data+'sit_cs2wfa/'+str(y)+'/CS2WFA_25km_'+str(y)+str(m).zfill(2)+'.nc'; print(filename)
                data = xr.open_dataset(filename,group='sea_ice_thickness')
                if k==1:
                  sitc = data.variables['sea_ice_thickness'][0]; #vdatac = data.variables[varray]#['sit']
                  sitc=np.where(sitc>0,sitc, np.nan); sitc=np.where(sitc<10,sitc, np.nan)
                  #if vname[0:12]=='sit_obs_rmse':
                  sitc=func.interp_field(sitc)
                  sitc=sitc.reshape(1,np.shape(sitc)[0],np.shape(sitc)[1])
                else:
                  sit = data.variables['sea_ice_thickness'][0]; sit=np.where(sit>0,sit, np.nan); sit=np.where(sit<10,sit, np.nan)
                  #if vname[0:12]=='sit_obs_rmse':
                  sit=func.interp_field(sit)
                  sit=sit.reshape(1,np.shape(sit)[0],np.shape(sit)[1])
                  sitc = np.concatenate([sitc,sit],0) # 'time')
                timec.append(datetime.datetime(y,m,1))
                data.close()
              sit_obs=sitc
              time_ob=dates.date2num(timec); time_obss=dates.num2date(time_ob); time_obsd=pd.DatetimeIndex(time_obss)

          vdatac=np.where(vdatac!=0,vdatac,np.nan)

          # computing monthly mean prior to rmse
          if vname[0:12]=='sit_obs_rmse': # _diff' or _rmse
            sit_mod=vdatac
            km=-1; time=[]
            sicc_mod = np.zeros([ym_end-ym_start,np.shape(sit_mod)[1],np.shape(sit_mod)[2]])
            for ym in range( ym_start, ym_end ):
              km+=1; y, m = divmod( ym, 12 ); m+=1
              print(run+': computing monthly mean for '+str(y)+'/'+str(m).zfill(2))
              iyear=time_modd.year==y
              imonth=time_modd.month==m; iym=np.where(iyear*imonth==True)
              time.append(time_mods[iym[0][0]])
              sicc_mod[km]=np.nanmean(sit_mod[iyear*imonth],axis=0) # month average
              sicc_mod[km]=np.where(sicc_mod[km]>0,sicc_mod[km] , np.nan)
              sicc_mod[km]=np.where(sicc_mod[km]<10,sicc_mod[km] , np.nan)
            vdatac=sicc_mod
            time_mod=dates.date2num(time); time_mods=dates.num2date(time_mod); time_modd=pd.DatetimeIndex(time_mods)
            #exit()

          # loop in the four seasons
          km=-1; tseason=['JFM','ASO','JAS','OND']
          for m in [1,4]:
            km+=1; 
            kmm+=1; 
            print(run+': computing 3-month mean starting in month '+str(m).zfill(2))
            if m==1:
              imonth1o=time_obsd.month==1; imonth2o=time_obsd.month==2; imonth3o=time_obsd.month==3; 
              imonth1=time_modd.month==1; imonth2=time_modd.month==2; imonth3=time_modd.month==3; 
            if m==4: 
              imonth1o=time_obsd.month==8; imonth2o=time_obsd.month==9; imonth3o=time_obsd.month==10; 
              imonth1=time_modd.month==8; imonth2=time_modd.month==9; imonth3=time_modd.month==10; 
            if m==7: 
              imonth1o=time_obsd.month==7; imonth2o=time_obsd.month==8; imonth3o=time_obsd.month==9; 
              imonth1=time_modd.month==7; imonth2=time_modd.month==8; imonth3=time_modd.month==9; 
            if m==10: 
              imonth1o=time_obsd.month==10; imonth2o=time_obsd.month==11; imonth3o=time_obsd.month==12; 
              imonth1=time_modd.month==10; imonth2=time_modd.month==11; imonth3=time_modd.month==12; 

            sit_obss=np.concatenate((sit_obs[imonth1o],sit_obs[imonth2o],sit_obs[imonth3o]),0)
            vdatas=np.concatenate((vdatac[imonth1],vdatac[imonth2],vdatac[imonth3]),0)

            if vname[0:12]=='sit_obs_rmse':
              # loop for each grid point
              mean = np.zeros([np.shape(vdatas)[1],np.shape(vdatas)[2]]); mean[:]=np.nan
              for i in range(np.shape(vdatas)[1]):
                for ii in range(np.shape(vdatas)[2]):
                  sit_mo=vdatas[::,i,ii]; sit_ob=sit_obss[::,i,ii]; 
                  mean[i,ii]=np.sqrt(np.nanmean(np.square(np.subtract(sit_ob,sit_mo))))
            elif vname=='sit':
              mobs=np.mean(sit_obss,0)
              mobs=np.where(mobs!=0.0,mobs,np.nan)
              #mean=np.where(mean!=0.0,mean,np.nan)
              vdatas=np.where(vdatas!=0.0,vdatas,np.nan)
              mean=np.nanmean(vdatas,0)
              #exit()
            else:
              mobs=np.nanmean(sit_obss,0)
              mobs=np.where(mobs!=0.0,mobs,np.nan)
              mean=np.nanmean(vdatas,0)
              mean=np.where(mean!=0.0,mean,np.nan)

            if vname=='sit_obs_diff':
            #  mobs=func.interp_field(mobs)
              mean=mean-mobs

            # computing difference between 2 expts
            if vname=='sit_diff' or vname=='sit_obs_rmse_diff':
              if ex==expt[-1]:
                mean=(means[km,:,:]-mean)#*90
              else:
                if km==0:
                  means=np.zeros((4,np.shape(mean)[0],np.shape(mean)[1]))
                means[km,:,:]=mean

            if ex==ex: # pt[-1]:
              ax=fig.add_subplot(2,2,kmm+1)
              bm = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l')#,ax=ax[km])
              bm.drawcoastlines(linewidth=.5)
              bm.fillcontinents(color='grey',lake_color='aqua')
              # add wrap-around point in longitude.
              #mean,lon_obs = addcyclic (mean,lon_obs)
              longr, latgr = bm([0,0],[-90,-70.5])#,inverse=True)
              bm.plot(longr,latgr,color='grey',linewidth=2)
              #bm.drawparallels(np.arange(-90,-30,5))
              #bm.drawmeridians(np.arange(0,360,30))
              lonp, latp = bm(lon_mod,lat_mod)#,inverse=True)
              if vname=='sit': 
                plt.title(tseason[km]+' '+run+' '+vname,loc='center')
                cmap = cmocean.cm.dense_r
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=0,vmax=3.0)#,vmin=0,vmax=.015)
              elif vname=='sit_obs_rmse': 
                plt.title(tseason[km]+' '+run+' rmse',loc='center')
                cmap = cmocean.cm.amp
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=0,vmax=2.0)#,vmin=0,vmax=.015)
              elif vname=='sit_obs_diff':
                plt.title(tseason[km]+' '+runs[ex]+' - Obs.',loc='center')
                cmap = cmocean.cm.balance
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=-2.,vmax=2.)
              elif vname=='sit_obs_rmse_diff':
                plt.title(tseason[km]+' '+runs[ex]+' - '+runs[expt[1]]+' rmse',loc='center')
                cmap = cmocean.cm.balance
                im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=-2.,vmax=2.)
              # contour
              lone, late = bm(lon_etopo,lat_etopo)#,inverse=True)
              ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
              clevels=[-800] # np.linspace(0,40,40,endpoint=False)
              ic=bm.contour(lonp,latp,h_etopoi,clevels,colors=('magenta'),linewidths=(.5,),origin='upper',linestyles='solid',extent=ext)
              #ic.clabel(clevels,fmt='%2.1f',colors='w',fontsize=20)
              # computing stats per subregion
              lon_regions=[-150,-61,-20,34,90,160]; 
              lat_regions=[ -77,-75,-73,-68.5,-67,-70];
              # text_map_w_stats(ax,data,lon_mod,bm,lon_regions,lat_regions,latn,oper,unit,colort):
              text_map_w_stats(ax,mean,lon_mod,bm,lon_regions,lat_regions,-60.0,'mean','m','black')
              mobs=mean
              plt.annotate('Total RMSE mean: '+format(np.nanmean(mobs),'.2f')+' m', xy=(.26,.56), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
              dataf=np.where(h_etopoi>=-800,mobs,np.nan); dataf=format(np.nanmean(dataf),'.2f')
              plt.annotate('Coastal RMSE mean: '+dataf+' m', xy=(.26,.51), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',
              dataf=np.where(h_etopoi<-800,mobs,np.nan); dataf=format(np.nanmean(dataf),'.2f')
              plt.annotate('Deep RMSE mean: '+dataf+' m', xy=(.26,.46), xycoords='axes fraction',fontsize=9,fontweight='bold')#, textcoords='offset points',

              # including colorbar
              divider = make_axes_locatable(ax)
              cax = divider.append_axes('right', size='5%', pad=0.05)
              fig.colorbar(im1, cax=cax, orientation='vertical')

          run='paper_1'
          figname=path_fig+run+'/map_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'

        elif vname[0:6]=='newice': # newice, newice_diff, 
          # loop in the four seasons
          km=-1; tseason=['JFM','AMJ','JAS','OND']
          for m in [1,4,7,10]:
            km+=1; 
            print(run+': computing 3-month mean starting in month '+str(m).zfill(2))
            if m==1:
              imonth1=time_modd.month==1; imonth2=time_modd.month==2; imonth3=time_modd.month==3; 
            if m==4: 
              imonth1=time_modd.month==4; imonth2=time_modd.month==5; imonth3=time_modd.month==6; 
            if m==7: 
              imonth1=time_modd.month==7; imonth2=time_modd.month==8; imonth3=time_modd.month==9; 
            if m==10: 
              imonth1=time_modd.month==10; imonth2=time_modd.month==11; imonth3=time_modd.month==12; 

            vdatas=np.concatenate((vdatac[imonth1],vdatac[imonth2],vdatac[imonth3]),0)
            mean=np.nanmean(vdatas,0)*90
            mean=np.where(mean!=0.0,mean,np.nan)

            # computing difference between 2 expts
            if vname=='newice_diff':
              if ex==expt[-1]:
                mean=(means[km,:,:]-mean)#*90
              else:
                if km==0:
                  means=np.zeros((4,np.shape(mean)[0],np.shape(mean)[1]))
                means[km,:,:]=mean

            if ex==expt[-1]:
              ax=fig.add_subplot(2,2,km+1)
              bm = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l')#,ax=ax[km])
  
              bm.drawcoastlines(linewidth=.5)
              bm.fillcontinents(color='grey',lake_color='aqua')
              # add wrap-around point in longitude.
              #mean,lon_obs = addcyclic (mean,lon_obs)
              longr, latgr = bm([0,0],[-90,-70.5])#,inverse=True)
              bm.plot(longr,latgr,color='grey',linewidth=2)
              #bm.drawparallels(np.arange(-90,-30,5))
              #bm.drawmeridians(np.arange(0,360,30))
              lonp, latp = bm(lon_mod,lat_mod)#,inverse=True)
              if vname[0:6]=='newice': # newice, newice_diff, 
                if len(expt)==1:
                  plt.title(tseason[km]+' '+run+' newice',loc='center')
                  cmap = cmocean.cm.matter
                  im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=0,vmax=.6)#,vmin=0,vmax=.015)
                else:
                  plt.title(tseason[km]+' '+runs[expt[0]]+' - '+runs[expt[1]],loc='center')
                  cmap = cmocean.cm.balance
                  im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=-.2,vmax=.2)
              # contour
              lone, late = bm(lon_etopo,lat_etopo)#,inverse=True)
              ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
              clevels=[-300] # np.linspace(0,40,40,endpoint=False)
              ic=bm.contour(lonp,latp,h_etopoi,clevels,colors=('grey'),linewidths=(.5,),origin='upper',linestyles='solid',extent=ext)
              #ic.clabel(clevels,fmt='%2.1f',colors='w',fontsize=20)
              # computing stats per subregion
              lon_regions=[-150,-61,-20,34,90,160];
              lat_regions=[ -77,-75,-73,-68.5,-67,-70];
              text_map_w_stats(mean,lon_mod,bm,lon_regions,lat_regions,'sum','m')
              plt.annotate('Total int. diff.: '+format(np.nansum(mean),'.2f')+' m', xy=(.3,.56), xycoords='axes fraction',fontsize=9)#, textcoords='offset points',
              dataf=np.where(h_etopoi>=-300,mean,np.nan); dataf=format(np.nansum(dataf),'.2f')
              plt.annotate('Coastal int. diff.: '+dataf+' m', xy=(.3,.51), xycoords='axes fraction',fontsize=9)#, textcoords='offset points',
              dataf=np.where(h_etopoi<-300,mean,np.nan); dataf=format(np.nansum(dataf),'.2f')
              plt.annotate('Deep int. diff.: '+dataf+' m', xy=(.3,.46), xycoords='axes fraction',fontsize=9)#, textcoords='offset points',
              # including colorbar
              divider = make_axes_locatable(ax)
              cax = divider.append_axes('right', size='5%', pad=0.05)
              fig.colorbar(im1, cax=cax, orientation='vertical')

          figname=path_fig+run+'/map_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'

        fig.tight_layout() 
        if ex==expt[-1]:
          if save_fig==1:
            #run='paper_1'
            if os.path.exists(path_fig+run)==False:
              os.mkdir(path_fig+run)
            print('Saving: '+figname)
            plt.savefig(figname,dpi=300,bbox_inches='tight')
          if plt_show==1:
            plt.show()
    
    
      ### Plot video 
      if plot_video==1:
        print('Ploting video: '+vname+' '+run)
        plt.rcParams.update({'font.size': 12})
        if ke==1 and (vname=='sie' or vname=='sic'): # if first expt load obs
          fig, ax = plt.subplots(1,len(obs_sources)+len(expt)+len(expt), figsize = (16,8)) # landscape
          # plot obs
          if vname=='sie' or vname=='sic': # sea ice extent
            # loop in time to read obs
            kc=0; ll=[]
            for obs_source in obs_sources: 
              ll.append('OBS-'+obs_source); k=0; kc+=1
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:12]=='OSISAF-ease2':
                #if obs_source[0:11]=='OSISAF-ease':
                #  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                if obs_source[0:12]=='OSISAF-ease2':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease2-250_icdr-v2p0_20180101.nc';
                data = xr.open_dataset(file)
                lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
                xobs = data.variables['xc']; yobs = data.variables['yc']
                data.close()
                dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
                st = tictoc.time();   print('Creating weights to interp. obs to model grid ...'); # get the start time
                func=myInterp.IrregularGridInterpolator(np.array(lon_obs),np.array(lat_obs),np.array(lon_nex),np.array(lat_nex))#[0]
                et = tictoc.time()-st; print('Execution time:', et, 'seconds')
              for t in time_obs:
                k+=1
                if obs_source=='NSIDC':
                  file=path_data+'/sic_nsidc/'+t.strftime("%Y")+'/'+'seaice_conc_daily_sh__'+t.strftime("%Y%m%d")+'_f17_v04r00.nc'
                  print(file)
                  obs_grid_area=25
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc']
                    #exit()
                  else:
                    sic_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'tdim' )
                elif obs_source[0:6]=='OSISAF':
                  #if obs_source[0:11]=='OSISAF-ease':
                  #  file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  if obs_source[0:12]=='OSISAF-ease2':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease2-250_icdr-v2p0_'+t.strftime("%Y%m%d")+'.nc'; 
                  else:
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_polstere-100_multi_'+t.strftime("%Y%m%d")+'.nc'
                    obs_grid_area=12.53377297 # 10 polstere
                  print(file)
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc']
                  else:
                    sic_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'time' )
                data.close()
    
              print('Processing obs SIC to get extent')
              if interp_obs==1:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_nex)[0],np.shape(lon_nex)[1]])
              else:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_obs)[0],np.shape(lon_obs)[1]])
              for t in range(np.shape(sicc_obs)[0]):
                sicct=sicc_obs[t]; 
                if interp_obs==1:
                  st = tictoc.time();   print('Interping obs to model grid ...'); # get the start time
                  sicobsi=func.interp_field(np.array(sicct))#[0]
                  # fixing gap due to interp method 
                  for tt in range(0,150): #226,np.shape(sicc_mod)[1]):  
                    sicobsi[tt][150]=sicobsi[tt][151] 
                  #sicobsi=seapy.oasurf(np.array(lon_obs),np.array(lat_obs),np.array(sicct),np.array(lon_nex),np.array(lat_nex))[0]
                  et = tictoc.time()-st; print('Execution time:', et, 'seconds')
                  sicct=sicobsi

                if vname=='sie':
                  siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
                  #iext=np.where(sicct>1); sicct[iext]=0;
                  #iext=np.where(sicct>.15)[0]; sicct[iext]=1;
                  iext=np.where(sicct>.15); 
                  st = tictoc.time(); print('Processing obs SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM")) # get the start time
                  for ii in range(np.shape(iext)[1]):
                    siccz[iext[0][ii],iext[1][ii]]=1.
                  #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
                  siccz[inan_mod]=np.nan
                  sic_obs[t]=siccz
                else:
                  sicct[inan_mod]=np.nan
                  sic_obs[t]=sicct
    
              sicc_obs=sic_obs
    
            plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
      
            time=time_obs; mask=1-mask; 
            variable = sicc_obs;
            interval=10 #len(time))
      
            #fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))
            ax[ke-1].set_title(obs_source,loc='right')
            ax[ke-1].set_title('Date : {} '.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
            ax[ke-1].set_facecolor('xkcd:putty')
      
            # including colorbar
            divider = make_axes_locatable(ax[ke-1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cmap = cmocean.cm.ice
            if vname=='sic':
              cmap = cmocean.cm.ice
            elif vname=='sit':
              cmap = cmocean.cm.dense_r
            im1 = ax[ke-1].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmin=0,vmax = 1.0)
            fig.colorbar(im1, cax=cax, orientation='vertical')
      
        if ke>=1 and (vname=='sie' or vname=='sic'): # if first expt load obs
          fps=24
          sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
    
          diff=np.abs(int(time_obsni[0])-np.array(time_modi)); min_diff=np.min(diff)
          ifirst=np.where(diff==min_diff)[0][0]#-1; 
          if ifirst<0:
            ifirst=0
          diff=np.abs(int(time_obsni[-1])-np.array(time_modi)); min_diff=np.min(diff)
          ilast=np.where(diff==min_diff)[-1][-1]+1
          sicc_mo=np.zeros((len(time_mod[ifirst:ilast])+1,np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
          #sicc_mo=sic_mod[ifirst:ilast+1][:][:] #,np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
          k=-1
          #Processing model SIC to get extent 
          for t in range(ifirst,ilast+1,1): # (np.shape(sicc_mod)[0]):
            k+=1
            print('Processing model SIC to get extent time: '+time_mods[t].strftime("%Y%m%d%HH:%MM"))
            sicct=sic_mod[t];
            #iext=np.where(sicct>.15)#[0]; sicct[iext]=1;
            #for ii in range(np.shape(iext)[1]):
            #  sicct[iext[0][ii],iext[1][ii]]=1.
            sicc_mo[k]=sicct
  
          if run=='BSOSE':
            st = tictoc.time();   print('Creating weights to interp. BSOSE to model grid ...'); # get the start time
            func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_nex),np.array(lat_nex))#[0]
            et = tictoc.time()-st; print('Execution time:', et, 'seconds')
  
          time_modi=time_modi[ifirst:ilast]
          # daily average
          if interp_obs==1:
            sicc_mod=np.zeros((len(time_obs),np.shape(lon_nex)[0],np.shape(lon_nex)[1]))
          else:
            sicc_mod=np.zeros((len(time_obs),np.shape(lon_obs)[0],np.shape(lon_obs)[1]))
          iday2=-9999
          for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
            if run=='BSOSE':
              # find the closest date
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              print(iday)
              if iday!=iday2:
                # interp to nextsim grid
                siccz=func.interp_field(np.array(sicc_mo[iday]))#[0]
                #st = tictoc.time(); print('Interp BSOSE SIC to nextsim grid: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                #siccz=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sicc_mo[iday]),np.array(lon_nex),np.array(lat_nex))[0]
                #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
                siccz[inan_mod]=np.nan
                iday2=iday;
    
              print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              sicc_ex=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
              iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
              for ii in range(np.shape(iext)[1]):
                sicc_ex[iext[0][ii],iext[1][ii]]=1.
              sicc_mod[t]=sicc_ex # np.nanmean(sicc_mo[iday,:,:],axis=0)
    
            else:  
              iday=np.where(time_obsni[t]==time_modi)[0]
              if interp_obs==1:
                siccz=np.nanmean(sicc_mo[iday,:,:],axis=0)
                #exit()
                #sicc_mod[t]=siccz # np.nanmean(sicc_mo[iday,:,:],axis=0)
                if vname=='sie':
                  print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                  sicct=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
                  iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
                  for ii in range(np.shape(iext)[1]):
                    sicct[iext[0][ii],iext[1][ii]]=1.
                  sicct[inan_mod]=np.nan
                  sicc_mod[t]=sicct
                else:
                  siccz[inan_mod]=np.nan
                  sicc_mod[t]=siccz
              else:
                sicc_modm=np.nanmean(sicc_mo[iday,:,:],axis=0)
                st = tictoc.time(); print('Interp model SIC to obs grid: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                sicc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sicc_modm),np.array(lon_obs),np.array(lat_obs))[0]
                et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
          #sit_output = datac.sit.to_masked_array() # Extract a given variable
          #time = datac.time.indexes['time']
      
          time=time_mods; mask=1-mask; 
          variable = sicc_mod;
          interval=10 #len(time))
      
          #fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))
          ax[ke].set_title(run,loc='right')
          #ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke].set_facecolor('xkcd:putty')
          # including colorbar
          divider = make_axes_locatable(ax[ke])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          cmap = cmocean.cm.ice
          im2 = ax[ke].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmin=0.0,vmax = 1.0)
          fig.colorbar(im2, cax=cax, orientation='vertical')
      
          #plotting the difference
          sicc_diff=sicc_mod-sicc_obs
          ax[ke+1].set_title('Mod-Obs',loc='right')
          #ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke+1].set_facecolor('xkcd:putty')
          # including colorbar
          divider = make_axes_locatable(ax[ke+1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          cmap = cmocean.cm.balance
          im3 = ax[ke+1].imshow(sicc_diff[0],cmap=cmap,origin = 'lower',animated=True,vmin=-1.0,vmax=1.0)
          fig.colorbar(im3, cax=cax, orientation='vertical')
    
        if ke==1 and vname=='sit': # if first expt load obs
          fig, ax = plt.subplots(1,len(obs_sources)+len(expt)+len(expt), figsize = (16,8)) # landscape
          #fig = plt.figure(figsize=(16,6),dpi=150)
    
          # Loading data
          k=0
          fps=24
          filename=path_data+'sit_cs2wfa/'+str(2015)+'/CS2WFA_25km_'+str(2015)+'0'+str(1)+'.nc'

          data = xr.open_dataset(filename)
          lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
          lon_obs=np.where(lon_obs<180,lon_obs,lon_obs-360)
          #lon_obs=np.where(lon_obs!=np.max(lon_obs),lon_obs,180.01)
          #lon_obs=np.where(lon_obs!=np.min(lon_obs),lon_obs,-180.01)
          sitc_obs = np.zeros([ym_end-ym_start,np.shape(sicc)[1],np.shape(sicc)[2]])
          for ym in range( ym_start, ym_end ):
            k+=1
            y, m = divmod( ym, 12 ); m+=1
            filename=path_data+'sit_cs2wfa/'+str(y)+'/CS2WFA_25km_'+str(y)+str(m).zfill(2)+'.nc'
            print(filename)
            data = xr.open_dataset(filename,group='sea_ice_thickness')
            if k==1:
              #datac = data.variable[vname];    timec = data.variables['time']; 
              sitc = data.variables['sea_ice_thickness']; #vdatac = data.variables[varray]#['sit']
            else:
              #datac = xr.concat([datac,data],'time'); time = data.variables['time'];   timec = xr.Variable.concat([timec,time],'time')
              sit = data.variables['sea_ice_thickness'];  sitc = xr.Variable.concat([sitc,sit],'time')
              #vdata = data.variables[varray]# ['sit']; 
              #vdatac = xr.Variable.concat([vdatac,vdata],'time')
            data.close()
    
          sicc_obs=sitc
          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
     
          # no latlon 
          #ax[ke-1]=fig.add_subplot(1,len(obs_sources)+len(expt)+len(expt), ke ,projection=proj_info.crs)
          # guillaume ax=fig.add_subplot(1,len(obs_sources)+len(expt)+len(expt), ke ,projection=proj_info.crs)
          ax[ke-1].set_title('CS2WFA',loc='right')
          #ax[ke-1].set_title('Date : {} '.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke-1].set_title(''.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
          cmap = cmocean.cm.dense_r
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke-1])
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          # add wrap-around point in longitude.
          #mean,lon_obs = addcyclic (mean,lon_obs)
          longr, latgr = m([0,0],[-90,-70.5])#,inverse=True)
          m.plot(longr,latgr,color='grey',linewidth=2)
          lonp, latp = m(lon_obs,lat_obs)#,inverse=True)
          im1 = m.pcolormesh(lonp,latp,sicc_obs[0],cmap=cmap,vmin=0,vmax = 3.5)
          #Madrid; x,y = m([-3.703889],[40.4125]); m.plot(x,y, marker="o", color="blue", label="Madrid", ls="")
          # including colorbar
          divider = make_axes_locatable(ax[ke-1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im1, cax=cax, orientation='vertical')
    
        if ke>=1 and vname=='sit': # if first expt load obs
          fps=1
          sit_mod = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          sicc_mod = np.zeros([ym_end-ym_start,np.shape(sicc_obs)[1],np.shape(sicc_obs)[2]])
          st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
          func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
          et = tictoc.time()-st; print('Execution time:', et, 'seconds')
          km=-1; time=[]
          for ym in range( ym_start, ym_end ):
            km+=1; y, m = divmod( ym, 12 ); m+=1
            print(run+': computing monthly mean for '+str(y)+'/'+str(m).zfill(2))
            iyear=time_modd.year==y
            imonth=time_modd.month==m; iym=np.where(iyear*imonth==True)
            time.append(time_mods[iym[0][0]])
            sit_modm=np.nanmean(sit_mod[iyear*imonth],axis=0) # month average
            #st = tictoc.time();   print('Interping model to obs grid ...'); # get the start time
            sicc_mod[km]=func.interp_field(np.array(sit_modm))#[0]
            #sicc_mod[km]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sit_modm),np.array(lon_obs),np.array(lat_obs))[0]
            sicc_mod[km]=np.where(sicc_mod[km]>0,sicc_mod[km] , np.nan)
            sicc_mod[km]=np.where(sicc_mod[km]<10,sicc_mod[km] , np.nan)
            #f=interpolate.RectBivariateSpline(np.array(lon_mod),np.array(lat_mod),np.array(sit_modm))
            #sicc_mod[km]=f(np.array(lon_obs),np.array(lat_obs))
            #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          time_modm=time; time_obs=time; 
          variable = sicc_mod;
          interval=10 #len(time))
      
          ##fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))
          ##ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          #ax[ke].set_facecolor('xkcd:putty')
          ## including colorbar
          #divider = make_axes_locatable(ax[ke])
          #cax = divider.append_axes('right', size='5%', pad=0.05)
          #cmap = cmocean.cm.dense_r
          #im2 = ax[ke].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmin=0,vmax = 3.5)
          #fig.colorbar(im2, cax=cax, orientation='vertical')
          
          # fixing gap due to interp method 
          for t in range(np.shape(sicc_mod)[0]):  
            for tt in range(226,np.shape(sicc_mod)[1]):  
              sicc_mod[t][tt][158]=sicc_mod[t][tt][157] 
    
          #ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke].set_title(run,loc='right')
          cmap = cmocean.cm.dense_r
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke])
          lonp, latp = m(lon_obs,lat_obs)#,inverse=True)
          im2 = m.pcolormesh(lonp,latp,sicc_mod[0],cmap=cmap,vmin=0,vmax = 3.5)
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          divider = make_axes_locatable(ax[ke])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im2, cax=cax, orientation='vertical')
          
          #plotting the difference
          sicc_diff=sicc_mod-sicc_obs
          ax[ke+1].set_title('Mod-Obs',loc='right')
          cmap = cmocean.cm.balance
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke+1])
          lonp, latp = m(lon_obs,lat_obs)#,inverse=True)
          im3 = m.pcolormesh(lonp,latp,sicc_diff[0],cmap=cmap,vmin=-2,vmax=2)
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          divider = make_axes_locatable(ax[ke+1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im3, cax=cax, orientation='vertical')
    
        # Ice drift
        if ke==1 and vname=='drift': # if first expt load obs
          fig, ax = plt.subplots(1,3, figsize = (16,8)) # landscape
          k=0
          for t in time_obs:
            k+=1 # drift_osisaf_ease2
            file=path_data+'/drift_osisaf_ease2/'+t.strftime("%Y")+'/ice_drift_sh_ease2-750_cdr-v1p0_24h-'+t.strftime("%Y%m%d")+'1200.nc'; 
            print(file)
            data = xr.open_dataset(file)
            if k==1:
              u_obs = data.variables['dX'] #['cdr_seaice_conc']
              v_obs = data.variables['dY'] #['cdr_seaice_conc']
              uc_obs = data.variables['dX'] #['cdr_seaice_conc']
              vc_obs = data.variables['dY'] #['cdr_seaice_conc']
              lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
              v_spao=2
              lon_obsv=lon_obs[::v_spao,::v_spao]
              lat_obsv=lat_obs[::v_spao,::v_spao]
            else:
              u_obs = data.variables['dX'] #['cdr_seaice_conc']
              v_obs = data.variables['dY'] #['cdr_seaice_conc']
              uc_obs = xr.Variable.concat([uc_obs,u_obs] ,'time' )
              vc_obs = xr.Variable.concat([vc_obs,v_obs] ,'time' )
            data.close()
    
          magc_obs=np.sqrt(uc_obs**2+vc_obs**2)
          uc_obs=uc_obs[:,::v_spao,::v_spao]
          vc_obs=vc_obs[:,::v_spao,::v_spao]
          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
          fps=24
     
          ax[ke-1].set_title('OSI-455',loc='right')
          ax[ke-1].set_title(time[0].strftime('%Y.%m.%d'), loc = 'left')
          cmap = cmocean.cm.tempo
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke-1])
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          lonop, latop = m(lon_obs,lat_obs)#,inverse=True)
          lonov, latov = m(lon_obsv,lat_obsv)#,inverse=True)
          im1 = m.pcolormesh(lonop,latop,magc_obs[0],cmap=cmap,vmin=0,vmax =80)
          # including colorbar
          divider = make_axes_locatable(ax[ke-1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im1, cax=cax, orientation='vertical')
          im11 = m.quiver(lonov, latov, uc_obs[0], vc_obs[0],color='black',width=0.002,scale=500.0) 
          qk=plt.quiverkey(im11,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
    
        if ke>=1 and vname=='drift': # if first expt load obs
          u_mod = udatac*3.6*24;  v_mod = vdatac*3.6*24;
          #u_mod=np.where(u_mod!=0.0,u_mod,np.nan)
          #v_mod=np.where(v_mod!=0.0,v_mod,np.nan)
          uc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); uc_mod[:]=np.nan
          vc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); vc_mod[:]=np.nan
          st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
          func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
          et = tictoc.time()-st; print('Execution time:', et, 'seconds')
  
          # daily data (average if nextsim)
          iday2=-9999
          for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
            if run=='BSOSE':
              print('BSOSE day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              print(iday)
              if iday!=iday2:
                uc_modi=func.interp_field(np.array(u_mod[iday]))
                vc_modi=func.interp_field(np.array(v_mod[iday]))
                iday2=iday;
              uc_mod[t]=uc_modi
              vc_mod[t]=vc_modi
            else:
              print('Computing model drift daily mean: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              iday=np.where(time_obsni[t]==time_modi)[0]
              ucmod=np.nanmean(u_mod[iday,:,:],axis=0)
              vcmod=np.nanmean(v_mod[iday,:,:],axis=0)
              uc_mod[t]=func.interp_field(np.array(ucmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
              vc_mod[t]=func.interp_field(np.array(vcmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
              #st = tictoc.time();   print('Interping model siu and siv to obs grid ...'); # get the start time
              #uc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(ucmod),np.array(lon_obs),np.array(lat_obs))[0]
              #vc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(vcmod),np.array(lon_obs),np.array(lat_obs))[0]
              #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          uc_mod=np.where(uc_mod!=0.0,uc_mod,np.nan)
          vc_mod=np.where(vc_mod!=0.0,vc_mod,np.nan)
          time=time_obs; 
          magc_mod=np.sqrt(uc_mod**2+vc_mod**2)
          magc_mod=np.where(magc_mod<=80.0,magc_mod,np.nan)
          uc_mod=np.where(magc_mod<=80.0,uc_mod,np.nan)
          vc_mod=np.where(magc_mod<=80.0,vc_mod,np.nan)
          uc_mod=uc_mod[:,::v_spao,::v_spao]
          vc_mod=vc_mod[:,::v_spao,::v_spao]
          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
          fps=6

          if ke==1: # first experiment
            uc_mod1=uc_mod; vc_mod1=vc_mod; magc_mod1=magc_mod

          ax[ke].set_title(run,loc='right')
          cmap = cmocean.cm.tempo
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke])
          lonp, latp = m(lon_mod,lat_mod)#,inverse=True)
          lonv, latv = m(lon_modv,lat_modv)#,inverse=True)
          if ke==1: # first experiment
            im2 = m.pcolormesh(lonop,latop,magc_mod[0],cmap=cmap,vmin=0,vmax=80)
          else:
            im3 = m.pcolormesh(lonop,latop,magc_mod[0],cmap=cmap,vmin=0,vmax=80)
          divider = make_axes_locatable(ax[ke])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          if ke==1: # first experiment
            fig.colorbar(im2, cax=cax, orientation='vertical')
            im22 = m.quiver(lonov, latov, uc_mod[0], vc_mod[0],color='black',width=0.002,scale=500.0) 
            qk=plt.quiverkey(im22,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
          else:
            fig.colorbar(im3, cax=cax, orientation='vertical')
            im33 = m.quiver(lonov, latov, uc_mod[0], vc_mod[0],color='black',width=0.002,scale=500.0) 
            qk=plt.quiverkey(im33,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          # add wrap-around point in longitude.
          #mean,lon_obs = addcyclic (mean,lon_obs)
          longr, latgr = m([0,0],[-90,-70.5])#,inverse=True)
          m.plot(longr,latgr,color='grey',linewidth=2)

          if len(expt)==1: # one expt only
            #plotting the difference
            ax[ke+1].set_title('Mod-Obs',loc='right')
            cmap = cmocean.cm.balance
            m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke+1])
            im3 = m.pcolormesh(lonop,latop,magc_mod[0]-magc_obs[0],cmap=cmap,vmin=-50,vmax=50)
            m.drawcoastlines()
            m.fillcontinents(color='grey',lake_color='aqua')
            divider = make_axes_locatable(ax[ke+1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im3, cax=cax, orientation='vertical')

          fig.tight_layout()

        # Ice divergence
        if vname=='divergence' or vname=='shear' or vname=='newice': # if first expt load obs
          if ke==1:
            fig, ax = plt.subplots(1,len(expt), figsize = (16,8)) # landscape
            time_obs=time_obsix

          if vname=='divergence' or vname=='shear':# or vname=='newice': # if first expt load obs
            u_mod = udatac*3.6*24;  v_mod = vdatac*3.6*24;
            #u_mod=np.where(u_mod!=0.0,u_mod,np.nan)
            #v_mod=np.where(v_mod!=0.0,v_mod,np.nan)
          elif vname=='newice': # if first expt load obs
            div_mod=vdatac
            div_mod=np.where(div_mod!=0.0,div_mod,np.nan)
          
          uc_mod = np.zeros([len(time_obsix),np.shape(vdatac)[1],np.shape(vdatac)[2]]); uc_mod[:]=np.nan
          vc_mod = np.zeros([len(time_obsix),np.shape(vdatac)[1],np.shape(vdatac)[2]]); vc_mod[:]=np.nan
          #st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
          #func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
          #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
  
          # daily data (average if nextsim)
          #iday2=-9999
          if vname!='newice':
            for t in range(len(time_obsix)): # (np.shape(sicc_mod)[0]):
              if run=='BSOSE':
                print('BSOSE day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
                iday=np.where(diff==min_diff)[0][0];
                print(iday)
                if iday!=iday2:
                  uc_modi=func.interp_field(np.array(u_mod[iday]))
                  vc_modi=func.interp_field(np.array(v_mod[iday]))
                  iday2=iday;
                uc_mod[t]=uc_modi
                vc_mod[t]=vc_modi
              else:
                #print('Cocatenating model drift results: '+time_obsix[t].strftime("%Y%m%d%HH:%MM"))
                iday=np.where(time_obsixn[t]==time_mod)[0]
                uc_mod[t]=u_mod[iday,:,:]
                vc_mod[t]=v_mod[iday,:,:]
                #uc_mod[t]=func.interp_field(np.array(ucmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
                #vc_mod[t]=func.interp_field(np.array(vcmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
                #st = tictoc.time();   print('Interping model siu and siv to obs grid ...'); # get the start time
                #uc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(ucmod),np.array(lon_obs),np.array(lat_obs))[0]
                #vc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(vcmod),np.array(lon_obs),np.array(lat_obs))[0]
                #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
            uc_mod=np.where(uc_mod!=0.0,uc_mod,np.nan)
            vc_mod=np.where(vc_mod!=0.0,vc_mod,np.nan)
            time=time_obsix; 

            dudx=(uc_mod[::,::,1::]-uc_mod[::,::,0:-1])/25.
            dvdx=(vc_mod[::,::,1::]-vc_mod[::,::,0:-1])/25.
            dudy=(uc_mod[::,1::,::]-uc_mod[::,0:-1,::])/25.
            dvdy=(vc_mod[::,1::,::]-vc_mod[::,0:-1,::])/25.
          
          if vname=='divergence': # 
            div_mod=dudx[::,0:-1,::]+dvdy[::,::,0:-1]
          elif vname=='shear': # shear
            div_mod=np.sqrt( np.square(dudx[::,0:-1,::]+dvdy[::,::,0:-1]) + np.square(dudy[::,0::,1::]+dvdx[::,1::,0::]) )

          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
          fps=24
    
          if vname=='divergence': # if first expt load obs
            cmap = cmocean.cm.balance
            vmin=-.2;vmax=.2
          elif vname=='shear': # if first expt load obs
            cmap = cmocean.cm.thermal_r
            vmin=0;vmax=.2
          elif vname=='newice': # if first expt load obs
            cmap = cmocean.cm.matter
            vmin=0;vmax=.03

          ax[ke-1].set_title(run,loc='right')
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke-1])
          if vname=='newice': # if first expt load obs
            lonp, latp = m(lon_mod,lat_mod) #,inverse=True)
            lonv, latv = m(lon_modv,lat_modv)#,inverse=True)
          else:
            lonp, latp = m(lon_mod[0:-1,0:-1],lat_mod[0:-1,0:-1]) #,inverse=True)
            lonv, latv = m(lon_modv,lat_modv)#,inverse=True)
          divider = make_axes_locatable(ax[ke-1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          if ke==1:
            div_mo=div_mod
            im1 = m.pcolormesh(lonp,latp,div_mo[0],cmap=cmap,vmin=vmin,vmax=vmax)
            fig.colorbar(im1, cax=cax, orientation='vertical')
          else:
            im2 = m.pcolormesh(lonp,latp,div_mod[0],cmap=cmap,vmin=vmin,vmax=vmax)
            fig.colorbar(im2, cax=cax, orientation='vertical')
          #im22 = m.quiver(lonov, latov, uc_mod[0], vc_mod[0],color='black',width=0.002,scale=500.0) 
          #qk=plt.quiverkey(im22,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          # add wrap-around point in longitude.
          longr, latgr = m([0,0],[-90,-70.5])#,inverse=True)
          m.plot(longr,latgr,color='grey',linewidth=2)
 
        if vname=='drift':
          if len(expt)==1:
            def animates(i):
              im1.set_array(magc_obs[i])
              im11.set_UVC(uc_obs[i],vc_obs[i], C=None)
              #plt.title('Date :{} '.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
              ax[0].set_title('{}'.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
              im2.set_array(magc_mod[i])
              im22.set_UVC(uc_mod[i],vc_mod[i], C=None)
              im3.set_array(magc_mod[i]-magc_obs[i])
              return [im1,im11,im2,im22,im3]
          else:
            def animates(i):
              im1.set_array(magc_obs[i])
              im11.set_UVC(uc_obs[i],vc_obs[i], C=None)
              #plt.title('Date :{} '.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
              ax[0].set_title('{}'.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
              im2.set_array(magc_mod1[i])
              im22.set_UVC(uc_mod1[i],vc_mod1[i], C=None)
              im3.set_array(magc_mod[i])
              im33.set_UVC(uc_mod[i],vc_mod[i], C=None)
              return [im1,im11,im2,im22,im3,im33]
    
        elif vname=='divergence' or vname=='shear' or vname=='newice':
          if ex==expt[-1]:
            def animates(i):
              im1.set_array(div_mo[i])
              ax[0].set_title('{}'.format(time_obsix[i].strftime('%Y.%m.%d')), loc = 'left')
              im2.set_array(div_mod[i])
              return [im1,im2]
    
        else: # obs x mod x diff
          def animates(i):
            im1.set_array(sicc_obs[i])
            #plt.title('Date :{} '.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
            ax[0].set_title('{}'.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
            im2.set_array(sicc_mod[i])
            im3.set_array(sicc_diff[i])
            return [im1,im2,im3]
    
        if ex==expt[-1]:
          time=time_obs
          #Nt = np.shape(variable)[0]
          plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
          anim = animation.FuncAnimation(fig, animates, frames=len(time),
                                             interval=interval, blit=True)
          FFwriter = animation.FFMpegWriter( fps = fps)
          if plt_show==1:
            plt.show()

          ##Save animation 
          if vname=='divergence':
            figname=path_fig+run+'/video_map_mod_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.mp4'
          else:
            figname=path_fig+run+'/video_map_mod_x_obs_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.mp4'

          if save_fig==1:
            print('Saving: '+figname)
            if os.path.exists(path_fig+run)==False:
              os.mkdir(path_fig+run)
            anim.save(figname, writer=FFwriter, dpi = 150)
    
    
      ### Plot scatter plot
      if plot_scatter==1:
        print('Ploting scatter: '+vname+' '+run)
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(1, 1, figsize = (8,8))
        if vname=='sst':
          sst = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
          sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          sic=sic.where(sic > .15 , np.nan)
          plt.plot(sst[0],sic[0],'.b')
          plt.xlabel('SST (oC)'); plt.ylabel('SIT (m)'); plt.title('SST x SIC (>.15)')
          figname=path_fig+run+'/sst_x_sic_'+str(start_year)+'-'+str(start_month)+'_'+str(end_year)+'-'+str(end_month)+'.png'
      
      ### Make animation of solo model output
      if plot_anim==1:
        print('Ploting anim: '+vname+' '+run)
    
        fps=24
        plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
        time = time_modd # datac.time.indexes['time']
      
        variable = vdatac;
        variable=np.where(variable!=0,variable,np.nan)
        interval=10 #len(time))
      
        fig, ax = plt.subplots(1, 1 ,figsize=(8,8))
        ax.set_title(run+' '+vname,loc='right')
        ax.set_title('Date : {} '.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
      
        if vname=='sic':
          cmap = cmocean.cm.ice; vmin=0.0; vmax=1.
        elif vname=='sit':
          cmap = cmocean.cm.dense_r; vmin=0; vmax=3.5
        elif vname=='newice':
          cmap = cmocean.cm.matter; vmin=0; vmax=0.03

        ax.set_title(run,loc='right')
        m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax)
        lonp, latp = m(lon_mod,lat_mod)#,inverse=True)
        lonv, latv = m(lon_modv,lat_modv)#,inverse=True)
        im1 = m.pcolormesh(lonp, latp, variable[0], cmap=cmap, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        #im22 = m.quiver(lonov, latov, uc_mod[0], vc_mod[0],color='black',width=0.002,scale=500.0) 
        #qk=plt.quiverkey(im22,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
        m.drawcoastlines()
        m.fillcontinents(color='grey',lake_color='aqua')
      
        def animate(i):
            im1.set_array(variable[i])
            ax.set_title('Date :{} '.format(time[i].strftime('%Y.%m.%d')), loc = 'left')
            return [im1]
      
        Nt = np.shape(variable)[0]
        anim = animation.FuncAnimation(fig, animate, frames=len(time),
                                         interval=interval, blit=True)
      
        #plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
        #sit_output = vdatac # .sit.to_masked_array() # Extract a given variable
        #time = time_modd # datac.time.indexes['time']
      
        #variable = sit_output;
        #anim=make_animation_util(time=time , mask =1- mask, variable = sit_output,interval=10)#len(time))
      
        FFwriter = animation.FFMpegWriter(fps=fps)
        ##Save animation 
        figname=path_fig+run+'/video_solo_map_'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.mp4'
        if save_fig==1:
          if os.path.exists(path_fig+run)==False:
            os.mkdir(path_fig+run)
          print('Saving: '+figname)
          anim.save(figname, writer=FFwriter, dpi = 150)


      ### Plot OBS and MODEL maps 
      if plot_mapo==1:
        print('Ploting obs and model maps: '+vname+' '+run)
        plt.rcParams.update({'font.size': 12})
        if ke==1 and (vname=='sie' or vname=='sic'): # if first expt load obs
          fig, ax = plt.subplots(1,len(obs_sources)+len(expt)+len(expt), figsize = (16,8)) # landscape
          # plot obs
          if vname=='sie' or vname=='sic': # sea ice extent
            # loop in time to read obs
            kc=0; ll=[]
            for obs_source in obs_sources: 
              ll.append('OBS-'+obs_source); k=0; kc+=1
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:12]=='OSISAF-ease2':
                #if obs_source[0:11]=='OSISAF-ease':
                #  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                if obs_source[0:12]=='OSISAF-ease2':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease2-250_icdr-v2p0_20180101.nc';
                data = xr.open_dataset(file)
                lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
                xobs = data.variables['xc']; yobs = data.variables['yc']
                data.close()
                dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
                st = tictoc.time();   print('Creating weights to interp. obs to model grid ...'); # get the start time
                func=myInterp.IrregularGridInterpolator(np.array(lon_obs),np.array(lat_obs),np.array(lon_nex),np.array(lat_nex))#[0]
                et = tictoc.time()-st; print('Execution time:', et, 'seconds')
              for t in time_obs:
                k+=1
                if obs_source=='NSIDC':
                  file=path_data+'/sic_nsidc/'+t.strftime("%Y")+'/'+'seaice_conc_daily_sh__'+t.strftime("%Y%m%d")+'_f17_v04r00.nc'
                  print(file)
                  obs_grid_area=25
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc']
                    #exit()
                  else:
                    sic_obs = data.variables['nsidc_nt_seaice_conc']#['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'tdim' )
                elif obs_source[0:6]=='OSISAF':
                  #if obs_source[0:11]=='OSISAF-ease':
                  #  file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  if obs_source[0:12]=='OSISAF-ease2':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease2-250_icdr-v2p0_'+t.strftime("%Y%m%d")+'.nc'; 
                  else:
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_polstere-100_multi_'+t.strftime("%Y%m%d")+'.nc'
                    obs_grid_area=12.53377297 # 10 polstere
                  print(file)
                  data = xr.open_dataset(file)
                  if k==1:
                    sicc_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc']
                  else:
                    sic_obs = data.variables['ice_conc']/100. #['cdr_seaice_conc'];  
                    sicc_obs = xr.Variable.concat([sicc_obs,sic_obs] ,'time' )
                data.close()
    
              print('Processing obs SIC to get extent')
              if interp_obs==1:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_nex)[0],np.shape(lon_nex)[1]])
              else:
                sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_obs)[0],np.shape(lon_obs)[1]])
              for t in range(np.shape(sicc_obs)[0]):
                sicct=sicc_obs[t]; 
                if interp_obs==1:
                  st = tictoc.time();   print('Interping obs to model grid ...'); # get the start time
                  sicobsi=func.interp_field(np.array(sicct))#[0]
                  # fixing gap due to interp method 
                  for tt in range(0,150): #226,np.shape(sicc_mod)[1]):  
                    sicobsi[tt][150]=sicobsi[tt][151] 
                  #sicobsi=seapy.oasurf(np.array(lon_obs),np.array(lat_obs),np.array(sicct),np.array(lon_nex),np.array(lat_nex))[0]
                  et = tictoc.time()-st; print('Execution time:', et, 'seconds')
                  sicct=sicobsi

                if vname=='sie':
                  siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
                  #iext=np.where(sicct>1); sicct[iext]=0;
                  #iext=np.where(sicct>.15)[0]; sicct[iext]=1;
                  iext=np.where(sicct>.15); 
                  st = tictoc.time(); print('Processing obs SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM")) # get the start time
                  for ii in range(np.shape(iext)[1]):
                    siccz[iext[0][ii],iext[1][ii]]=1.
                  #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
                  siccz[inan_mod]=np.nan
                  sic_obs[t]=siccz
                else:
                  sicct[inan_mod]=np.nan
                  sic_obs[t]=sicct
    
              sicc_obs=sic_obs
    
            plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
      
            time=time_obs; mask=1-mask; 
            variable = sicc_obs;
            interval=10 #len(time))
      
            #fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))
            ax[ke-1].set_title(obs_source,loc='right')
            ax[ke-1].set_title('Date : {} '.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
            ax[ke-1].set_facecolor('xkcd:putty')
      
            # including colorbar
            divider = make_axes_locatable(ax[ke-1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cmap = cmocean.cm.ice
            if vname=='sic':
              cmap = cmocean.cm.ice
            elif vname=='sit':
              cmap = cmocean.cm.dense_r
            im1 = ax[ke-1].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmin=0,vmax = 1.0)
            fig.colorbar(im1, cax=cax, orientation='vertical')
      
        if ke>=1 and (vname=='sie' or vname=='sic'): # if first expt load obs
          fps=24
          sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
    
          diff=np.abs(int(time_obsni[0])-np.array(time_modi)); min_diff=np.min(diff)
          ifirst=np.where(diff==min_diff)[0][0]#-1; 
          if ifirst<0:
            ifirst=0
          diff=np.abs(int(time_obsni[-1])-np.array(time_modi)); min_diff=np.min(diff)
          ilast=np.where(diff==min_diff)[-1][-1]+1
          sicc_mo=np.zeros((len(time_mod[ifirst:ilast])+1,np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
          #sicc_mo=sic_mod[ifirst:ilast+1][:][:] #,np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
          k=-1
          #Processing model SIC to get extent 
          for t in range(ifirst,ilast+1,1): # (np.shape(sicc_mod)[0]):
            k+=1
            print('Processing model SIC to get extent time: '+time_mods[t].strftime("%Y%m%d%HH:%MM"))
            sicct=sic_mod[t];
            #iext=np.where(sicct>.15)#[0]; sicct[iext]=1;
            #for ii in range(np.shape(iext)[1]):
            #  sicct[iext[0][ii],iext[1][ii]]=1.
            sicc_mo[k]=sicct
  
          if run=='BSOSE':
            st = tictoc.time();   print('Creating weights to interp. BSOSE to model grid ...'); # get the start time
            func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_nex),np.array(lat_nex))#[0]
            et = tictoc.time()-st; print('Execution time:', et, 'seconds')
  
          time_modi=time_modi[ifirst:ilast]
          # daily average
          if interp_obs==1:
            sicc_mod=np.zeros((len(time_obs),np.shape(lon_nex)[0],np.shape(lon_nex)[1]))
          else:
            sicc_mod=np.zeros((len(time_obs),np.shape(lon_obs)[0],np.shape(lon_obs)[1]))
          iday2=-9999
          for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
            if run=='BSOSE':
              # find the closest date
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              print(iday)
              if iday!=iday2:
                # interp to nextsim grid
                siccz=func.interp_field(np.array(sicc_mo[iday]))#[0]
                #st = tictoc.time(); print('Interp BSOSE SIC to nextsim grid: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                #siccz=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sicc_mo[iday]),np.array(lon_nex),np.array(lat_nex))[0]
                #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
                siccz[inan_mod]=np.nan
                iday2=iday;
    
              print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              sicc_ex=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
              iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
              for ii in range(np.shape(iext)[1]):
                sicc_ex[iext[0][ii],iext[1][ii]]=1.
              sicc_mod[t]=sicc_ex # np.nanmean(sicc_mo[iday,:,:],axis=0)
    
            else:  
              iday=np.where(time_obsni[t]==time_modi)[0]
              if interp_obs==1:
                siccz=np.nanmean(sicc_mo[iday,:,:],axis=0)
                #exit()
                #sicc_mod[t]=siccz # np.nanmean(sicc_mo[iday,:,:],axis=0)
                if vname=='sie':
                  print('Processing model SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                  sicct=np.zeros((np.shape(siccz)[0],np.shape(siccz)[1]))
                  iext=np.where(siccz>.15)#[0]; sicct[iext]=1;
                  for ii in range(np.shape(iext)[1]):
                    sicct[iext[0][ii],iext[1][ii]]=1.
                  sicct[inan_mod]=np.nan
                  sicc_mod[t]=sicct
                else:
                  siccz[inan_mod]=np.nan
                  sicc_mod[t]=siccz
              else:
                sicc_modm=np.nanmean(sicc_mo[iday,:,:],axis=0)
                st = tictoc.time(); print('Interp model SIC to obs grid: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
                sicc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sicc_modm),np.array(lon_obs),np.array(lat_obs))[0]
                et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
          #sit_output = datac.sit.to_masked_array() # Extract a given variable
          #time = datac.time.indexes['time']
      
          time=time_mods; mask=1-mask; 
          variable = sicc_mod;
          interval=10 #len(time))
      
          #fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))
          ax[ke].set_title(run,loc='right')
          #ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke].set_facecolor('xkcd:putty')
          # including colorbar
          divider = make_axes_locatable(ax[ke])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          cmap = cmocean.cm.ice
          im2 = ax[ke].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmin=0.0,vmax = 1.0)
          fig.colorbar(im2, cax=cax, orientation='vertical')
      
          #plotting the difference
          sicc_diff=sicc_mod-sicc_obs
          ax[ke+1].set_title('Mod-Obs',loc='right')
          #ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke+1].set_facecolor('xkcd:putty')
          # including colorbar
          divider = make_axes_locatable(ax[ke+1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          cmap = cmocean.cm.balance
          im3 = ax[ke+1].imshow(sicc_diff[0],cmap=cmap,origin = 'lower',animated=True,vmin=-1.0,vmax=1.0)
          fig.colorbar(im3, cax=cax, orientation='vertical')
    
        if ke==1 and vname=='sit': # if first expt load obs
          fig, ax = plt.subplots(1,len(obs_sources)+len(expt)+len(expt), figsize = (16,8)) # landscape
          #fig = plt.figure(figsize=(16,6),dpi=150)
    
          # Loading data
          k=0
          fps=24
          filename=path_data+'sit_cs2wfa/'+str(2015)+'/CS2WFA_25km_'+str(2015)+'0'+str(1)+'.nc'

          data = xr.open_dataset(filename)
          lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
          lon_obs=np.where(lon_obs<180,lon_obs,lon_obs-360)
          #lon_obs=np.where(lon_obs!=np.max(lon_obs),lon_obs,180.01)
          #lon_obs=np.where(lon_obs!=np.min(lon_obs),lon_obs,-180.01)
          sitc_obs = np.zeros([ym_end-ym_start,np.shape(sicc)[1],np.shape(sicc)[2]])
          for ym in range( ym_start, ym_end ):
            k+=1
            y, m = divmod( ym, 12 ); m+=1
            filename=path_data+'sit_cs2wfa/'+str(y)+'/CS2WFA_25km_'+str(y)+str(m).zfill(2)+'.nc'
            print(filename)
            data = xr.open_dataset(filename,group='sea_ice_thickness')
            if k==1:
              #datac = data.variable[vname];    timec = data.variables['time']; 
              sitc = data.variables['sea_ice_thickness']; #vdatac = data.variables[varray]#['sit']
            else:
              #datac = xr.concat([datac,data],'time'); time = data.variables['time'];   timec = xr.Variable.concat([timec,time],'time')
              sit = data.variables['sea_ice_thickness'];  sitc = xr.Variable.concat([sitc,sit],'time')
              #vdata = data.variables[varray]# ['sit']; 
              #vdatac = xr.Variable.concat([vdatac,vdata],'time')
            data.close()
    
          sicc_obs=sitc
          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
     
          # no latlon 
          #ax[ke-1]=fig.add_subplot(1,len(obs_sources)+len(expt)+len(expt), ke ,projection=proj_info.crs)
          # guillaume ax=fig.add_subplot(1,len(obs_sources)+len(expt)+len(expt), ke ,projection=proj_info.crs)
          ax[ke-1].set_title('CS2WFA',loc='right')
          #ax[ke-1].set_title('Date : {} '.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke-1].set_title(''.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
          cmap = cmocean.cm.dense_r
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke-1])
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          # add wrap-around point in longitude.
          #mean,lon_obs = addcyclic (mean,lon_obs)
          longr, latgr = m([0,0],[-90,-70.5])#,inverse=True)
          m.plot(longr,latgr,color='grey',linewidth=2)
          lonp, latp = m(lon_obs,lat_obs)#,inverse=True)
          im1 = m.pcolormesh(lonp,latp,sicc_obs[0],cmap=cmap,vmin=0,vmax = 3.5)
          #Madrid; x,y = m([-3.703889],[40.4125]); m.plot(x,y, marker="o", color="blue", label="Madrid", ls="")
          # including colorbar
          divider = make_axes_locatable(ax[ke-1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im1, cax=cax, orientation='vertical')
    
        if ke>=1 and vname=='sit': # if first expt load obs
          fps=1
          sit_mod = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          sicc_mod = np.zeros([ym_end-ym_start,np.shape(sicc_obs)[1],np.shape(sicc_obs)[2]])
          st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
          func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
          et = tictoc.time()-st; print('Execution time:', et, 'seconds')
          km=-1; time=[]
          for ym in range( ym_start, ym_end ):
            km+=1; y, m = divmod( ym, 12 ); m+=1
            print(run+': computing monthly mean for '+str(y)+'/'+str(m).zfill(2))
            iyear=time_modd.year==y
            imonth=time_modd.month==m; iym=np.where(iyear*imonth==True)
            time.append(time_mods[iym[0][0]])
            sit_modm=np.nanmean(sit_mod[iyear*imonth],axis=0) # month average
            #st = tictoc.time();   print('Interping model to obs grid ...'); # get the start time
            sicc_mod[km]=func.interp_field(np.array(sit_modm))#[0]
            #sicc_mod[km]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(sit_modm),np.array(lon_obs),np.array(lat_obs))[0]
            sicc_mod[km]=np.where(sicc_mod[km]>0,sicc_mod[km] , np.nan)
            sicc_mod[km]=np.where(sicc_mod[km]<10,sicc_mod[km] , np.nan)
            #f=interpolate.RectBivariateSpline(np.array(lon_mod),np.array(lat_mod),np.array(sit_modm))
            #sicc_mod[km]=f(np.array(lon_obs),np.array(lat_obs))
            #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          time_modm=time; time_obs=time; 
          variable = sicc_mod;
          interval=10 #len(time))
      
          ##fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))
          ##ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          #ax[ke].set_facecolor('xkcd:putty')
          ## including colorbar
          #divider = make_axes_locatable(ax[ke])
          #cax = divider.append_axes('right', size='5%', pad=0.05)
          #cmap = cmocean.cm.dense_r
          #im2 = ax[ke].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmin=0,vmax = 3.5)
          #fig.colorbar(im2, cax=cax, orientation='vertical')
          
          # fixing gap due to interp method 
          for t in range(np.shape(sicc_mod)[0]):  
            for tt in range(226,np.shape(sicc_mod)[1]):  
              sicc_mod[t][tt][158]=sicc_mod[t][tt][157] 
    
          #ax[ke].set_title('Date : {} '.format(time_obs[0].strftime('%Y.%m.%d')), loc = 'left')
          ax[ke].set_title(run,loc='right')
          cmap = cmocean.cm.dense_r
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke])
          lonp, latp = m(lon_obs,lat_obs)#,inverse=True)
          im2 = m.pcolormesh(lonp,latp,sicc_mod[0],cmap=cmap,vmin=0,vmax = 3.5)
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          divider = make_axes_locatable(ax[ke])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im2, cax=cax, orientation='vertical')
          
          #plotting the difference
          sicc_diff=sicc_mod-sicc_obs
          ax[ke+1].set_title('Mod-Obs',loc='right')
          cmap = cmocean.cm.balance
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke+1])
          lonp, latp = m(lon_obs,lat_obs)#,inverse=True)
          im3 = m.pcolormesh(lonp,latp,sicc_diff[0],cmap=cmap,vmin=-2,vmax=2)
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
          divider = make_axes_locatable(ax[ke+1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im3, cax=cax, orientation='vertical')
    
        # Ice drift
        if ke==1 and vname=='drift': # if first expt load obs
          k=0
          for t in time_obs:
            k+=1 # drift_osisaf_ease2
            file=path_data+'/drift_osisaf_ease2/'+t.strftime("%Y")+'/ice_drift_sh_ease2-750_cdr-v1p0_24h-'+t.strftime("%Y%m%d")+'1200.nc'; 
            print(file)
            data = xr.open_dataset(file)
            if k==1:
              u_obs = data.variables['dX'] #['cdr_seaice_conc']
              v_obs = data.variables['dY'] #['cdr_seaice_conc']
              uc_obs = data.variables['dX'] #['cdr_seaice_conc']
              vc_obs = data.variables['dY'] #['cdr_seaice_conc']
              lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
              v_spao=2
              lon_obsv=lon_obs[::v_spao,::v_spao]
              lat_obsv=lat_obs[::v_spao,::v_spao]
            else:
              u_obs = data.variables['dX'] #['cdr_seaice_conc']
              v_obs = data.variables['dY'] #['cdr_seaice_conc']
              uc_obs = xr.Variable.concat([uc_obs,u_obs] ,'time' )
              vc_obs = xr.Variable.concat([vc_obs,v_obs] ,'time' )
            data.close()
    
          magc_obs=np.sqrt(uc_obs**2+vc_obs**2)
          uc_obs=uc_obs[:,::v_spao,::v_spao]
          vc_obs=vc_obs[:,::v_spao,::v_spao]
          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
          fps=24
    
        if ke>=1 and vname=='drift': # if first expt load obs
          u_mod = udatac*3.6*24;  v_mod = vdatac*3.6*24;
          #u_mod=np.where(u_mod!=0.0,u_mod,np.nan)
          #v_mod=np.where(v_mod!=0.0,v_mod,np.nan)
          uc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); uc_mod[:]=np.nan
          vc_mod = np.zeros([len(time_obs),np.shape(v_obs)[1],np.shape(v_obs)[2]]); vc_mod[:]=np.nan
          st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
          func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
          et = tictoc.time()-st; print('Execution time:', et, 'seconds')
  
          # daily data (average if nextsim)
          iday2=-9999
          for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
            if run=='BSOSE':
              print('BSOSE day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              diff=np.abs(int(time_obsni[t])-np.array(time_modi)); min_diff=np.min(diff)
              iday=np.where(diff==min_diff)[0][0];
              print(iday)
              if iday!=iday2:
                uc_modi=func.interp_field(np.array(u_mod[iday]))
                vc_modi=func.interp_field(np.array(v_mod[iday]))
                iday2=iday;
              uc_mod[t]=uc_modi
              vc_mod[t]=vc_modi
            else:
              print('Computing model drift daily mean: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              iday=np.where(time_obsni[t]==time_modi)[0]
              ucmod=np.nanmean(u_mod[iday,:,:],axis=0)
              vcmod=np.nanmean(v_mod[iday,:,:],axis=0)
              uc_mod[t]=func.interp_field(np.array(ucmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
              vc_mod[t]=func.interp_field(np.array(vcmod)) #,np.array(lon_obs),np.array(lat_obs))[0]
              #st = tictoc.time();   print('Interping model siu and siv to obs grid ...'); # get the start time
              #uc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(ucmod),np.array(lon_obs),np.array(lat_obs))[0]
              #vc_mod[t]=seapy.oasurf(np.array(lon_mod),np.array(lat_mod),np.array(vcmod),np.array(lon_obs),np.array(lat_obs))[0]
              #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
    
          uc_mod=np.where(uc_mod!=0.0,uc_mod,np.nan)
          vc_mod=np.where(vc_mod!=0.0,vc_mod,np.nan)
          time=time_obs; 
          magc_mod=np.sqrt(uc_mod**2+vc_mod**2)
          #magc_mod=np.where(magc_mod<=80.0,magc_mod,np.nan)
          #uc_mod=np.where(magc_mod<=80.0,uc_mod,np.nan)
          #vc_mod=np.where(magc_mod<=80.0,vc_mod,np.nan)
          uc_mod=uc_mod[:,::v_spao,::v_spao]
          vc_mod=vc_mod[:,::v_spao,::v_spao]
          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
          fps=6

          if ke==1:
            magc_mods=np.zeros((len(expt),np.shape(magc_mod)[0],np.shape(magc_mod)[1],np.shape(magc_mod)[2]))
            uc_mods=np.zeros((len(expt),np.shape(uc_mod)[0],np.shape(uc_mod)[1],np.shape(uc_mod)[2]))
            vc_mods=np.zeros((len(expt),np.shape(vc_mod)[0],np.shape(vc_mod)[1],np.shape(vc_mod)[2]))
          magc_mods[ke-1]=magc_mod
          uc_mods[ke-1]=uc_mod
          vc_mods[ke-1]=vc_mod

        # if expt is last plot
        if ex==expt[-1]:
          for t in range(len(time_obs)):
 
            fig, ax = plt.subplots(1,1+len(expt), figsize = (16,8)) # landscape

            for l in range(1+len(expt)): 

              if l==0: #obs
                magc=magc_obs; uc=uc_obs; vc=vc_obs
                lon=lon_obs; lat=lat_obs
                lonv=lon_obsv; latv=lat_obsv
                ax[l].set_title('Observed drift ',loc='left')
              else: # mod
                magc=magc_mods[l-1]; uc=uc_mods[l-1]; vc=vc_mods[l-1]
                #lon=lon_mod,lat=lat_mod
                #lonv=lon_modv,latv=lat_modv
                run=runs[expts[expt[l-1]]]
                ax[l].set_title(run+' drift (km/day)',loc='left')

              ax[l].set_title(time[0].strftime('%Y/%m/%d'), loc = 'right')
              cmap = cmocean.cm.tempo

              m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[l])
              m.drawcoastlines()
              m.fillcontinents(color='grey',lake_color='aqua')
              lonop, latop = m(lon,lat)#,inverse=True)
              lonov, latov = m(lonv,latv)#,inverse=True)
              im1 = m.pcolormesh(lonop,latop,magc[t],cmap=cmap,vmin=0,vmax =80)
              # including colorbar
              #divider = make_axes_locatable(ax[l])
              #cax = divider.append_axes('right', size='5%', pad=0.05)
              #fig.colorbar(im1, cax=cax, orientation='vertical')
              # add wrap-around point in longitude.
              longr, latgr = m([0,0],[-90,-70.5])#,inverse=True)
              m.plot(longr,latgr,color='grey',linewidth=2)
              im11 = m.quiver(lonov,latov,uc[t],vc[t],color='black',width=0.002,scale=500.0) 
              qk=plt.quiverkey(im11,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})

          fig.tight_layout()
          fig.subplots_adjust(right=0.825)
          cax = fig.add_axes([0.83, 0.238, 0.02, 0.525]) # [left, bottom, width, height]
          cbar=fig.colorbar(im1, cax=cax)
          cbar.ax.tick_params(labelsize=16)
          ##Save figure 
          figname=path_fig+run+'/map_obs_mod_'+vname+'_'+format(time[t].strftime('%Y_%m_%d_%H_%M'))+'.png'
          if save_fig==1:
            if os.path.exists(path_fig+run)==False:
              os.mkdir(path_fig+run)
            print('Saving: '+figname)
            plt.savefig(figname,dpi=300,bbox_inches='tight')
            plt.close('all') 
      
      ### Make plots of model(s) output
      if plot_smap==1:
        print('Ploting solo exp maps: '+vname+' '+run)
        font = {'weight' : 'bold',
                'size'   : 18}
        matplotlib.rc('font', **font)
    
        fps=24
        plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
        time = time_modd # datac.time.indexes['time']
      
          #mean,lon_obs = addcyclic (mean,lon_obs)

        if vname=='divergence':
          print('CHANGE dx IF YOU CHANGE MODEL RESOLUTION. run: '+run)
          uc_mod = udatac*3.6*24;  vc_mod = vdatac*3.6*24;
          dudx=(uc_mod[::,::,1::]-uc_mod[::,::,0:-1])/25.
          dvdx=(vc_mod[::,::,1::]-vc_mod[::,::,0:-1])/25.
          dudy=(uc_mod[::,1::,::]-uc_mod[::,0:-1,::])/25.
          dvdy=(vc_mod[::,1::,::]-vc_mod[::,0:-1,::])/25.
          div_mod=dudx[::,0:-1,::]+dvdy[::,::,0:-1]
          shear_mod=np.sqrt( np.square(dudx[::,0:-1,::]+dvdy[::,::,0:-1]) + np.square(dudy[::,0::,1::]+dvdx[::,1::,0::]) )
          lon_mod=lon_mod[0:-1,0:-1]; lat_mod=lat_mod[0:-1,0:-1]
          variable = div_mod;
          variable=np.where(variable!=0,variable,np.nan)
        else:
          variable = vdatac;
          variable=np.where(variable!=0,variable,np.nan)

        if vname=='sic':
          cmap = cmocean.cm.ice; vmin=0.8; vmax=1.
          vnamee='ice concentration '
        elif vname=='sit':
          cmap = cmocean.cm.dense_r; vmin=0; vmax=3.5
          vnamee='ice thickness (m) '
        elif vname=='newice':
          cmap = cmocean.cm.matter; vmin=0; vmax=0.03
          vnamee='ice growth (m/day) '
        elif vname=='divergence':
          cmap = cmocean.cm.balance; vmin=-0.2; vmax=0.2
          vnamee='divergence (1/day) '

        if ke==1:
          variables=np.zeros((len(expt),np.shape(variable)[0],np.shape(variable)[1],np.shape(variable)[2]))
        variables[ke-1]=variable

        # if expt is last plot
        if ex==expt[-1]:
          for ts in [len(time_obsixn)-1]:  
          #for ts in range(0,len(time_obsixn)):  
            ke=-1
            fig, ax = plt.subplots(1, len(expt) ,figsize=(16,8))
            for ex in expt:
              ke+=1
              variable=variables[ke]
              t=np.where(time_obsixn[ts]==time_mod)[0]; #exit()

              run=runs[expts[ex]]
              #if ex==expt[-1]:
              #  ax[ke].set_title(time[t].strftime('%Y/%m/%d %H:%M')[0], loc = 'right')
              #  ax[ke].set_title('Brittle model ice coverage',loc='center')
              #else:
              #  ax[ke].set_title('Viscous-elastic model ice coverage',loc='center')
              ax[ke].set_title(run+' '+vnamee,loc='left')

              m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke])
              lonp, latp = m(lon_mod,lat_mod)#,inverse=True)
              lonv, latv = m(lon_modv,lat_modv)#,inverse=True)
              im1 = m.pcolormesh(lonp, latp, variable[t[0]], cmap=cmap, vmin=vmin, vmax=vmax)
              divider = make_axes_locatable(ax[ke])
              #cax = divider.append_axes('right', size='5%', pad=0.05)
              #fig.colorbar(im1, cax=cax, orientation='vertical')
              #im22 = m.quiver(lonov, latov, uc_mod[0], vc_mod[0],color='black',width=0.002,scale=500.0) 
              #qk=plt.quiverkey(im22,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
              m.drawcoastlines()
              #m.drawparallels(np.arange(-90,-30,5))
              #m.drawmeridians(np.arange(0,360,30))

              m.fillcontinents(color='grey',lake_color='aqua')
              # add wrap-around point in longitude.
              longr, latgr = m([0,0],[-90,-70.5])#,inverse=True)
              m.plot(longr,latgr,color='grey',linewidth=2)

              # circle around the storm
              if ex==expt[-1]:
                longr, latgr = m([-33.0],[-69.5])#,inverse=True)
                #m.plot(longr,latgr,marker='o',color='magenta')#,markersize=4,linewidth=.02)
                m.scatter(longr,latgr,s=20000,facecolors='none', edgecolors='magenta',linewidth=2)
      
              #plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
              #sit_output = vdatac # .sit.to_masked_array() # Extract a given variable
              #time = time_modd # datac.time.indexes['time']
      
              #variable = sit_output;
              #anim=make_animation_util(time=time , mask =1- mask, variable = sit_output,interval=10)#len(time))
     
            ##Save figure 
            fig.tight_layout()
            fig.subplots_adjust(right=0.825)
            cax = fig.add_axes([0.85, 0.105, 0.02, 0.785]) # [left, bottom, width, height]
            cbar=fig.colorbar(im1, cax=cax)
            cbar.ax.tick_params(labelsize=16)
            figname=path_fig+run+'/map_solo_'+vname+'_'+format(time[t[0]].strftime('%Y_%m_%d_%H_%M'))+'.png'
            if save_fig==1:
              if os.path.exists(path_fig+run)==False:
                os.mkdir(path_fig+run)
              print('Saving: '+figname)
              plt.savefig(figname,dpi=300,bbox_inches='tight')
              plt.close('all') 
     
 
      
    
      #fig=plt.subplots(); plt.pcolormesh(datac.sit[0,:,:]); plt.colorbar(); plt.title(time[0]); plt.show()


print('End of script')
#plt.close('all')
