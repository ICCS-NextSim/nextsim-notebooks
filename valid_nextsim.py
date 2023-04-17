import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from netCDF4 import Dataset
from netCDF4 import date2num,num2date
import cmocean
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from matplotlib import dates
from mpl_toolkits.basemap import Basemap
#import cartopy
#import cartopy.crs as ccrs
import seapy
import irregular_grid_interpolator as myInterp
#from scipy import interpolate
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
start_day  =1
start_month=1
start_year =2016
end_day    =28
end_month  =12 # 
end_year   =2021


#Runs (names) or experiments (numbers - starts with 1)
exp=17
exptc=[12,9,exp,15]#2,5,7,10]
expt=exptc
expt=[18,19]

serie_or_maps=[0] # 1 for serie, 2 for video, 3 for map, 0 for neither
my_dates=1
inc_obs=1

#Variables
vname ='newice_diff' # sie,bsie,sit,sit_rmse,siv,drift,vcorr, processed variable e.g. 'bsie=(confusion matrix)', 'sit' 
                  # newice, newice_diff

# Plot types
plot_scatter=0
plot_series =0
plot_video  =0
plot_map    =1 # seasonal maps
plot_maps   =0
plot_anim   =0 # solo video
save_fig    =1
plt_show    =1
interp_obs  =1 # only for SIE maps obs has 2x the model resolution

####################################################################
# after BSOSE run (ocean boundary cond), runs are all mEVP
runs=['50km_ocean_wind'     ,'50km_bsose_20180102'   ,'50km_hSnowAlb_20180102','50km_61IceAlb_20180102','50km_14kPmax_20180102',
      '50km_20Clab_20180102','50km_P14C20_20180102'  ,'50km_LandNeg2_20180102','50km_bsose_20130102'   ,'50km_dragWat01_20180102',
      '50km_glorys_20180102','BSOSE'                 ,'50km_mevp_20130102'    ,'50km_lemieux_20130102' ,'50km_h50_20130102',           # 15
      '50km_hyle_20130102'  ,'50km_ckFFalse_20130102','BBM'                   ,'mEVP'] # last two are links to the original expts

#Colors
colors=['r','b','k','orange','pink','brown','yellow','g','r','b','k']
obs_colors=['g','y','orange'];

# varrays according to vname
if vname=='sic' or vname=='sie' or vname=='bsie':
  varray='sic' 
elif vname=='sit' or vname=='siv' or vname=='sit_rmse':
  varray='sit' 
elif vname=='drift' or vname=='vcorr' or vname=='divergence':
  varray='siv' 
elif vname=='newice' or vname=='newice_diff':
  varray='newice' 


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
obs_sources=['OSISAFease2']#,'OSISAF-ease'] #['NSIDC','OSISAF','OSISAF-ease','OSISAFease2']: 

#paths
print('Hostname: '+socket.gethostname())
if socket.gethostname()=='SC442555' or socket.gethostname()=='SC442555.local' or socket.gethostname()=='wifi-staff-172-24-40-164.net.auckland.ac.nz':
  path_runs='/Users/rsan613/n/southern/runs/' # ''~/'
  path_fig ='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/figures/'
  path_data ='/Users/rsan613/n/southern/data/'
  path_bsose='/Volumes/LaCie/mahuika/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/bsose/'
elif socket.gethostname()[0]=='w' or socket.gethostname()=='mahuika01' or socket.gethostname()=='mahuika':
  path_runs='/scale_wlg_persistent/filesets/project/uoa03669/rsan613/n/southern/runs/' # ''~/'
  path_fig ='/scale_wlg_persistent/filesets/project/uoa03669/rsan613/n/southern/figures/' 
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

if plot_map==1 and vname[0:6]=='newice': 
  #print('Interpolating etopo bathy to nextsim grid')
  #func=myInterp.IrregularGridInterpolator(np.array(lon_etopo),np.array(lat_etopo),np.array(lon_mod),np.array(lat_mod))
  #h_etopoi=func.interp_field(np.array(h_etopo))
  #h=xr.DataArray(h_etopoi) #,coords={'y': lat_e,'x': lon_e},dims=["y", "x"])
  filename=path_data+'etopo/ETOPO_Antarctic_50km_nextsim.nc'
  #h.to_netcdf(filename)
  print('Loading: '+filename)
  ds=xr.open_dataset(filename)
  h_etopoi=ds.variables['__xarray_dataarray_variable__'][:]


for serie_or_map in serie_or_maps:
  print(str(serie_or_map))

  # variables to be plotted vary with type (map or series)
  if serie_or_map==1: # series
    expt=exptc
    plot_series=1; plot_video=0; plot_map=0;
    vnames=[ 'sie','bsie','sit','siv','drift','vcorr'] # sie,bsie,sit,siv,drift,vcorr processed variable e.g. 'bsie=(confusion matrix)', 'sit' 
    varrays=['sic','sic' ,'sit','sit','siv' ,'siv'] # netcdf variable for each type of plot (raw variable used in xarray)
  elif serie_or_map==2: # video
    expt=[exp]
    plot_series=0; plot_video=1; plot_map=0;
    vnames=[ 'sie','sit','drift'] 
    varrays=['sic','sit','siv'  ] 
  elif serie_or_map==3: # map
    expt=[exp]
    plot_series=0; plot_video=0; plot_map=1;
    vnames=[ 'vcorr'] 
    varrays=['siv'] 
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

    timesix=pd.date_range(dates.num2date(time_ini), periods=int(time_fin-time_ini)*24/6, freq=('%dH' % int(6)))
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
              ll.append('OBS-'+obs_source); k=0; kc+=1
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:11]=='OSISAFease2':
                if obs_source[0:11]=='OSISAF-ease':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                elif obs_source[0:11]=='OSISAFease2':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease2-250_icdr-v2p0_20180101.nc';
                data = xr.open_dataset(file)
                xobs = data.variables['xc']; yobs = data.variables['yc']
                data.close()
                dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
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
                  if obs_source[0:11]=='OSISAF-ease':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  elif obs_source[0:11]=='OSISAFease2':
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
                siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
                #iext=np.where(sicct>1); sicct[iext]=0;
                #iext=np.where(sicct>.15)[0]; sicct[iext]=1;
                iext=np.where(sicct>.15); 
                for i in range(np.shape(iext)[1]):
                  siccz[iext[0][i],iext[1][i]]=1.
                #iext=np.where(sicct<=.15)[0]; sicct[iext]=0;
                #mean[t] = np.sum(sicct*25*25)
                if obs_source[0:11]=='OSISAF-ease' or obs_source[0:11]=='OSISAFease2':
                  meant = np.multiply(siccz[0:-1,0:-1],obs_grid_area); # meant = np.multiply(meant,obs_grid_area);
                else:
                  meant = np.multiply(siccz,obs_grid_area); meant = np.multiply(meant,obs_grid_area);
                mean[t] = np.sum(meant)
    
              plt.plot(time_obs, mean, color=obs_colors[kc-1])   
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
            figname=path_fig+run+'/serie_sit_domain_average_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
          elif inc_obs==1:
            if ke==1: # if first expt load obs
              kc=0; 
              # Loading data
              filename=path_data+'sit_cs2wfa/'+str(2020)+'/CS2WFA_25km_'+str(2020)+'0'+str(1)+'.nc'
              data = xr.open_dataset(filename); lon_obs = data.variables['lon']; lat_obs = data.variables['lat']
              lon_obs=np.where(lon_obs<180,lon_obs,lon_obs-360)
              lon_obs=np.where(lon_obs!=np.max(lon_obs),lon_obs,180)
              lon_obs=np.where(lon_obs!=np.min(lon_obs),lon_obs,-180)
              sitc_obs = np.zeros([ym_end-ym_start,np.shape(sicc)[1],np.shape(sicc)[2]])
              timec=[]
              if vname=='sit': 
                ll=['CS2WFA']
              else:
                ll=[]
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
                else:
                  sit = data.variables['sea_ice_thickness']; sit=np.where(sit>0,sit, np.nan); sit=np.where(sit<10,sit, np.nan)
                  sitc = np.concatenate([sitc,sit],0) # 'time')
                timec.append(datetime.datetime(y,m,1))
                data.close()
              sicc_obs=sitc
              if vname=='sit': 
                mean=np.nanmean(sicc_obs,axis=1); mean=np.nanmean(mean,axis=1)
                plt.plot(timec, mean, color=obs_colors[kc])   
                plt.grid('on')
    
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
              timec=time; 
              plt.ylabel('Sea ice thickness (m)'); plt.title('Sea ice thickness [Model interp to Obs]')
            elif vname=='sit_rmse':
              # masking mod where there is no obs
              sicc_mod=sicc_obs+(sicc_mod-sicc_obs)
              mean=np.zeros_like(time)
              print('Computing monthly thickness rmse')
              for t in range(0,len(time)):
                mean[t]=np.sqrt(np.nanmean(np.square(np.subtract(sicc_obs[t],sicc_mod[t]))))
              timec=time; 
              plt.ylabel('Sea ice thickness rmse (m)'); plt.title('Sea ice thickness rmse [Model interp to Obs]')

            figname=path_fig+run+'/serie_'+vname+'_month_mean_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
        elif vname=='sie':
          sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
          sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
          diff=np.abs(int(time_obsn[0])-np.array(time_modi)); min_diff=np.min(diff)
          ifirst=np.where(diff==min_diff)[0][0]-1; 
          if ifirst<0:
            ifirst=0
          diff=np.abs(int(time_obsn[-1])-np.array(time_modi)); min_diff=np.min(diff)
          ilast=np.where(diff==min_diff)[-1][-1]+1
          #ilast=np.where(int(time_obsn[-1])==time_modi)[0][-1]
          sicc_mo=np.zeros((len(time_mod[ifirst:ilast+1]),np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
          T=len(time_mod[ifirst:ilast])
          #T = np.shape(sit)[0]
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
    
          plt.ylabel('Sea ice extent (km\^2)'); plt.title('Sea ice extent [sum(area[sic>.15])]')
          figname=path_fig+run+'/serie_sie_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
          time=timec[ifirst:ilast] 
    
        elif vname=='bsie': # binary sea ice extent
          # loop in time to read obs
          if ke==1: # if first expt load obs
            kc=0; ll=[]
            for obs_source in obs_sources: 
              ll.append('OBS-'+obs_source); k=0; kc+=1
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:11]=='OSISAFease2':
                if obs_source[0:11]=='OSISAF-ease':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                elif obs_source[0:11]=='OSISAFease2':
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
                  if obs_source[0:11]=='OSISAF-ease':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  elif obs_source[0:11]=='OSISAFease2':
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
    
            diff=np.abs(int(time_obsn[0])-np.array(time_modi)); min_diff=np.min(diff)
            ifirst=np.where(diff==min_diff)[0][0]#-1; 
            if ifirst<0:
              ifirst=0
            diff=np.abs(int(time_obsn[-1])-np.array(time_modi)); min_diff=np.min(diff)
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
                diff=np.abs(int(time_obsn[t])-np.array(time_modi)); min_diff=np.min(diff)
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
                iday=np.where(time_obsn[t]==time_modi)[0]
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
            #plt.plot(time, mtotal, colors[ke-1],linestyle=':')   
            plt.plot(time, mneg, colors[ke-1],linestyle='--')   
            plt.plot(time, mpos, colors[ke-1],linestyle='-',marker='.')   
            plt.ylabel('(%)'); 
            plt.title('True positive (-), false positive (.-) and false negative (--) model-obs comparison')
            #plt.title('Accurate (-), over- (.-) and underestimated (--) ice coverage')
            figname=path_fig+run+'/serie_bsie_total_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
          
    
        elif vname=='siv':
          print('Plotting sea ice volume')
          if ke==1:
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
          figname=path_fig+run+'/serie_siv_total_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
        elif vname=='vcorr' or vname=='drift':
          if ke==1:
            if vname=='drift':
              ll=['OSI-455']; 
            else:
              ll=[]; 
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
    
            uc_obs=np.array(uc_obs); vc_obs=np.array(vc_obs)
            if vname=='drift':
              magc_obs=np.sqrt(uc_obs**2+vc_obs**2)
              mean=np.nanmean(magc_obs,1); mean=np.nanmean(mean,1)
              time=time_obs
              plt.plot(time, mean, obs_colors[ke-1])   
    
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
              diff=np.abs(int(time_obsn[t])-np.array(time_modi)); min_diff=np.min(diff)
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
              iday=np.where(time_obsn[t]==time_modi)[0]
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
              magc_mod=np.sqrt(uc_mod[t]**2+vc_mod[t]**2)
              #magc_mod=np.where(magc_mod<=80.0,magc_mod,np.nan)
              mean[t]=np.nanmean(magc_mod)
          if vname=='vcorr':
            plt.ylabel('Complex correlation'); plt.title('Ice drift complex correlation between model and obs (OSI-455)')
            plt.ylim([0,1]) 
            figname=path_fig+run+'/serie_vector_complex_correlation_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
          elif vname=='drift':
            plt.ylabel('Drift speed (km/day)'); plt.title('Antarctic sea-ice average drift speed (km/day)') 
            #plt.ylim([0,1]) 
            figname=path_fig+run+'/serie_velocity_speed_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'
    
          time=time_obs
      
        plt.plot(time, mean, colors[ke-1])   
        if ex==expt[-1]:
          for i in expt:
            ll.append(runs[i])
    
          plt.legend(ll)
          date_form = dates.DateFormatter("%b/%y")
          ax.xaxis.set_major_formatter(date_form)
          plt.tight_layout()
          if save_fig==1:
            if os.path.exists(path_fig+run)==False:
              os.mkdir(path_fig+run)
            print('Saving: '+figname)
            plt.savefig(figname)
          if plt_show==1:
            plt.show()
      
      ### Plot maps (seasonal) 
      if plot_map==1:
        print('Ploting map: '+vname+' '+run)
        plt.rcParams.update({'font.size': 12})
        fig=plt.figure(figsize = (9,8)) # landscape
        if vname=='vcorr': 
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
              diff=np.abs(int(time_obsn[t])-np.array(time_modi)); min_diff=np.min(diff)
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
              print('Computing model and obs vector complex correlation on day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              iday=np.where(time_obsn[t]==time_modi)[0]
              ucmod=np.nanmean(u_mod[iday,:,:],axis=0)
              vcmod=np.nanmean(v_mod[iday,:,:],axis=0)
              uc_mod[t]=func.interp_field(np.array(ucmod))
              vc_mod[t]=func.interp_field(np.array(vcmod))
              uc_mod[t]=np.where(uc_mod[t]!=0.0,uc_mod[t],np.nan)
              vc_mod[t]=np.where(vc_mod[t]!=0.0,vc_mod[t],np.nan)
          # loop in the four seasons
          km=-1; tseason=['JFM','AMJ','JAS','OND']
          for m in [1,4,7,10]:
            km+=1; 
            print(run+': computing seasonal complex vector correlation starting in month '+str(m).zfill(2))
            if m==1:
              imonth1=time_obsd.month==1; imonth2=time_obsd.month==2; imonth3=time_obsd.month==3; 
            if m==4: 
              imonth1=time_obsd.month==4; imonth2=time_obsd.month==5; imonth3=time_obsd.month==6; 
            if m==7: 
              imonth1=time_obsd.month==7; imonth2=time_obsd.month==8; imonth3=time_obsd.month==9; 
            if m==10: 
              imonth1=time_obsd.month==10; imonth2=time_obsd.month==11; imonth3=time_obsd.month==12; 
            umod=np.concatenate((uc_mod[imonth1],uc_mod[imonth2],uc_mod[imonth3]),0)
            vmod=np.concatenate((vc_mod[imonth1],vc_mod[imonth2],vc_mod[imonth3]),0)
            uobs=np.concatenate((uc_obs[imonth1],uc_obs[imonth2],uc_obs[imonth3]),0)
            vobs=np.concatenate((vc_obs[imonth1],vc_obs[imonth2],vc_obs[imonth3]),0)
    
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

    
            ax=fig.add_subplot(2,2,km+1)
            plt.title(tseason[km]+' '+run+' vel. corr.',loc='center')
            cmap = cmocean.cm.amp
            bm = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l')#,ax=ax[km])
            bm.drawcoastlines()
            bm.fillcontinents(color='grey',lake_color='aqua')
            lonp, latp = bm(lon_obs,lat_obs)#,inverse=True)
            im1 = bm.pcolormesh(lonp,latp,mean,cmap=cmap,vmin=0,vmax=1)
            # contour
            ext=[np.nanmin(lonp),np.nanmax(lonp),np.nanmin(latp),np.nanmax(latp)]
            clevels=[.7,1.]# np.linspace(0,40,40,endpoint=False)
            ic=bm.contour(lonp,latp,mean,clevels,colors=('k'),linewidths=(1.5,),origin='upper',linestyles='solid',extent=ext)
            #ic.clabel(clevels,fmt='%2.1f',colors='w',fontsize=20)
            # including colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')

          figname=path_fig+run+'/map_vector_complex_correlation_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.png'

        if vname[0:6]=='newice': 
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
              #bm.drawparallels(np.arange(-90,-30,5))
              #bm.drawmeridians(np.arange(0,360,30))
              lonp, latp = bm(lon_mod,lat_mod)#,inverse=True)
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
              ic=bm.contour(lonp,latp,h_etopoi,clevels,colors=('b'),linewidths=(1.5,),origin='upper',linestyles='solid',extent=ext)
              #ic.clabel(clevels,fmt='%2.1f',colors='w',fontsize=20)
              # computing stats per subregion
              lon_regions=[-150,-61,-20,34,90,160];
              text_map_w_stats(mean,lon_mod,bm,lon_regions,'sum','m')
              plt.annotate('Total int. diff.: '+format(np.nansum(mean),'.2f')+' m', xy=(.3,.56), xycoords='axes fraction',fontsize=9)#, textcoords='offset points',
              dataf=np.where(h_etopoi>=-300,mean,np.nan); dataf=format(np.nansum(dataf),'.2f')
              plt.annotate('Shallow int. diff.: '+dataf+' m', xy=(.3,.51), xycoords='axes fraction',fontsize=9)#, textcoords='offset points',
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
            if os.path.exists(path_fig+run)==False:
              os.mkdir(path_fig+run)
            print('Saving: '+figname)
            plt.savefig(figname)
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
              if obs_source[0:11]=='OSISAF-ease' or obs_source[0:11]=='OSISAFease2':
                if obs_source[0:11]=='OSISAF-ease':
                  file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
                elif obs_source[0:11]=='OSISAFease2':
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
                  if obs_source[0:11]=='OSISAF-ease':
                    file=path_data+'/sic_osisaf/'+t.strftime("%Y")+'/ice_conc_sh_ease-125_multi_'+t.strftime("%Y%m%d")+'.nc'; 
                  elif obs_source[0:11]=='OSISAFease2':
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
    
          diff=np.abs(int(time_obsn[0])-np.array(time_modi)); min_diff=np.min(diff)
          ifirst=np.where(diff==min_diff)[0][0]#-1; 
          if ifirst<0:
            ifirst=0
          diff=np.abs(int(time_obsn[-1])-np.array(time_modi)); min_diff=np.min(diff)
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
              diff=np.abs(int(time_obsn[t])-np.array(time_modi)); min_diff=np.min(diff)
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
              iday=np.where(time_obsn[t]==time_modi)[0]
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
          filename=path_data+'sit_cs2wfa/'+str(2020)+'/CS2WFA_25km_'+str(2020)+'0'+str(1)+'.nc'
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
          fig, ax = plt.subplots(1,len(obs_sources)+len(expt)+len(expt), figsize = (16,8)) # landscape
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
              diff=np.abs(int(time_obsn[t])-np.array(time_modi)); min_diff=np.min(diff)
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
              iday=np.where(time_obsn[t]==time_modi)[0]
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
    
          ax[ke].set_title(run,loc='right')
          cmap = cmocean.cm.tempo
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke])
          lonp, latp = m(lon_mod,lat_mod)#,inverse=True)
          lonv, latv = m(lon_modv,lat_modv)#,inverse=True)
          im2 = m.pcolormesh(lonop,latop,magc_mod[0],cmap=cmap,vmin=0,vmax=80)
          divider = make_axes_locatable(ax[ke])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im2, cax=cax, orientation='vertical')
          im22 = m.quiver(lonov, latov, uc_mod[0], vc_mod[0],color='black',width=0.002,scale=500.0) 
          qk=plt.quiverkey(im22,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
    
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

        # Ice divergence
        if vname=='divergence': # if first expt load obs
          if ke==1:
            fig, ax = plt.subplots(1,len(expt), figsize = (16,8)) # landscape
            time_obs=time_obsix

          u_mod = udatac*3.6*24;  v_mod = vdatac*3.6*24;
          #u_mod=np.where(u_mod!=0.0,u_mod,np.nan)
          #v_mod=np.where(v_mod!=0.0,v_mod,np.nan)
          
          uc_mod = np.zeros([len(time_obsix),np.shape(v_mod)[1],np.shape(v_mod)[2]]); uc_mod[:]=np.nan
          vc_mod = np.zeros([len(time_obsix),np.shape(v_mod)[1],np.shape(v_mod)[2]]); vc_mod[:]=np.nan
          #st = tictoc.time();   print('Creating weights to interp. model to obs grid ...'); # get the start time
          #func=myInterp.IrregularGridInterpolator(np.array(lon_mod),np.array(lat_mod),np.array(lon_obs),np.array(lat_obs))#[0]
          #et = tictoc.time()-st; print('Execution time:', et, 'seconds')
  
          # daily data (average if nextsim)
          #iday2=-9999
          for t in range(len(time_obsix)): # (np.shape(sicc_mod)[0]):
            if run=='BSOSE':
              print('BSOSE day: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
              diff=np.abs(int(time_obsn[t])-np.array(time_modi)); min_diff=np.min(diff)
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
          dvdy=(vc_mod[::,1::,::]-vc_mod[::,0:-1,::])/25.
          div_mod=dudx[::,0:-1,::]+dvdy[::,::,0:-1]

          time=time_obs; mask=1-mask; 
          interval=10 #len(time))
          fps=24
    
          cmap = cmocean.cm.balance
          ax[ke-1].set_title(run,loc='right')
          m = Basemap(projection='splaea',boundinglat=-55,lon_0=180,resolution='l',ax=ax[ke-1])
          lonp, latp = m(lon_mod[0:-1,0:-1],lat_mod[0:-1,0:-1]) #,inverse=True)
          lonv, latv = m(lon_modv,lat_modv)#,inverse=True)
          divider = make_axes_locatable(ax[ke-1])
          cax = divider.append_axes('right', size='5%', pad=0.05)
          if ke==1:
            div_mo=div_mod
            im1 = m.pcolormesh(lonp,latp,div_mo[0],cmap=cmap,vmin=-.2,vmax=.2)
            fig.colorbar(im1, cax=cax, orientation='vertical')
          else:
            im2 = m.pcolormesh(lonp,latp,div_mod[0],cmap=cmap,vmin=-.2,vmax=.2)
            fig.colorbar(im2, cax=cax, orientation='vertical')
          #im22 = m.quiver(lonov, latov, uc_mod[0], vc_mod[0],color='black',width=0.002,scale=500.0) 
          #qk=plt.quiverkey(im22,.5,.5,10,'10 km/day',labelpos='S',fontproperties={'size':8})
          m.drawcoastlines()
          m.fillcontinents(color='grey',lake_color='aqua')
    
    
        if vname=='drift':
          def animates(i):
            im1.set_array(magc_obs[i])
            im11.set_UVC(uc_obs[i],vc_obs[i], C=None)
            #plt.title('Date :{} '.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
            ax[0].set_title('{}'.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
            im2.set_array(magc_mod[i])
            im22.set_UVC(uc_mod[i],vc_mod[i], C=None)
            im3.set_array(magc_mod[i]-magc_obs[i])
            return [im1,im11,im2,im22,im3]
    
        elif vname=='divergence':
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
          cmap = cmocean.cm.ice; vmin=0; vmax=1.
        elif vname=='sit':
          cmap = cmocean.cm.dense_r; vmin=0; vmax=3.5
        elif vname=='newice':
          cmap = cmocean.cm.matter; vmin=0; vmax=0.04

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
      
      
    
      #fig=plt.subplots(); plt.pcolormesh(datac.sit[0,:,:]); plt.colorbar(); plt.title(time[0]); plt.show()


print('End of script')
plt.close('all')
