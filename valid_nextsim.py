import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from matplotlib import dates
import seapy
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utils import * #make_animation, time_series_plot, time_series_plot2
from sys import exit
import os
import socket
import time as tictoc
plt.ion()
plt.close('all')

#Time
start_day  =2
start_month=1
start_year =2019
end_day    =31
end_month  =12
end_year   =2019

#Runs (names) or experiments (numbers)
expt=[1]

# Plot types
plot_series=0
plot_map   =0
plot_video =1     
plot_anim  =0
save_fig   =1

#Colors
colors=['r','b','k']

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

####################################################################
expts=[0,1]
runs=['50km_ocean_wind','50km_bsose_20180102']


#Variables
vname='sie' #'sit' # timeseries
varim='sie' # 'sit' # video

#trick to cover all months in runs longer than a year
end_month=end_month+1
ym_start= 12*start_year + start_month - 1
ym_end  = 12*end_year + end_month - 1
end_month=end_month-1

#obs sources
obs_sources=['OSISAFease2']#,'OSISAF-ease'] #['NSIDC','OSISAF','OSISAF-ease','OSISAFease2']: 

#paths
if socket.gethostname()=='SC442555' or socket.gethostname()=='SC442555.local':
  path_runs='/Users/rsan613/n/southern/runs/' # ''~/'
  path_fig ='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/figures/'
  path_data ='/Users/rsan613/n/southern/data/'
else:
  print("Your runs, figures etc paths haven't been set")
  exit()
  
#Grid information
run=runs[expts[0]] # 'data_glorys'
data = xr.open_dataset(path_runs+run+'/output/Moorings_2018m01.nc')
lon_mod = data.longitude #sit.to_masked_array() # Extract a given variable
lat_mod = data.latitude #sit.to_masked_array() # Extract a given variable
sit_output = data.sit.to_masked_array() # Extract a given variable
inan_mod=ma.getmaskarray(sit_output[0]); 
mask = ma.getmaskarray(sit_output[0]) #Get mask

# time_obs
time_ini = dates.date2num(datetime.datetime(start_year,start_month,start_day))
time_fin = dates.date2num(datetime.datetime(end_year,end_month,end_day)) 
freqobs  = 1; # daily data
times=pd.date_range(dates.num2date(time_ini), periods=int(time_fin-time_ini)*freqobs, freq=('%dD' % int(1/freqobs)))
time_obsn=dates.date2num(times)
time_obs=dates.num2date(time_obsn)

#sic 
prefix_sic='seaice_conc_daily_sh__'; sufix_sic='_f17_v04r00.nc'

# Loop in the experiments
ke=0
for ex in expt:
  ke+=1
  run=runs[expts[ex]]

  # Loading data
  k=0
  for ym in range( ym_start, ym_end ):
    k+=1
    y, m = divmod( ym, 12 ); m+=1
    if m<10:
      filename=path_runs+run+'/output/Moorings_'+str(y)+'m0'+str(m)+'.nc'
    else:
      filename=path_runs+run+'/output/Moorings_'+str(y)+'m'+str(m)+'.nc'
    print(filename)
    data = xr.open_dataset(filename)
    if k==1:
      #datac = data.variable[vname]
      timec = data.variables['time']
      sicc = data.variables['sic']
      vdatac = data.variables['sit']
    else:
      #datac = xr.concat([datac,data],'time')
      time = data.variables['time'];   timec = xr.Variable.concat([timec,time],'time')
      sic = data.variables['sic'];   sicc = xr.Variable.concat([sicc,sic],'time')
      vdata = data.variables['sit']; vdatac = xr.Variable.concat([vdatac,vdata],'time')
      #exit() 
    data.close()

    time_mod=dates.date2num(timec)
    time_mods=dates.num2date(time_mod)
    time_modi=[int(time_mod[ii]) for ii in range(len(time_mod))] # integer time for daily search
  #datac.data_vars
  
  if plot_series==1:
    plt.rcParams.update({'font.size': 22})
    # Plotting time series
    if ke==1:
      fig, ax = plt.subplots(1, 1, figsize = (16,8)) # landscape
      # plot obs
      if vname=='sie': # sea ice extent
        # loop in time to read obs
        kc=0; obs_colors=['g','y','orange']; ll=[]
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

          print('Processing SIC to get extent')
          mean = np.zeros(np.shape(sicc_obs)[0])
          for t in range(np.shape(sicc_obs)[0]):
            print('Processing SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
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
 
    if vname=='sit':
      sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
      sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
      T = np.shape(sit)[0]
      mean = np.zeros(T)
      std = np.zeros(T)
      for t in range(T):
          #mean[t] = np.mean(variable[t][variable[t].mask == False])
          mean[t] = np.mean((sit[t]*sic[t])/sic[t])
      plt.ylabel('SIT (m)'); plt.title('Domain average sea ice thickness (SIT)')
      figname=path_fig+run+'/domain_average_sit_'+str(start_year)+'-'+str(start_month)+'_'+str(end_year)+'-'+str(end_month)+'.png'
    elif vname=='sie':
      sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
      sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
      T = np.shape(sit)[0]
      mean = np.zeros(T)
      std = np.zeros(T)
      for t in range(T):
        print('Processing model SIC to get extent time: '+time_mods[t].strftime("%Y%m%d%HH:%MM"))
        #mean[t] = np.sum(sic[t]*50*50)
        sicct=sic[t];
        siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
        #iext=np.where(sicct>1)[0]; sicct[iext]=0;
        iext=np.where(sicct>.15)#[0]; sicct[iext]=1;
        for i in range(np.shape(iext)[1]):
          siccz[iext[0][i],iext[1][i]]=1.
        #iext=np.where(sicct<=.15)[0]; sicct[iext]=0;
        meant = np.multiply(siccz,25); meant = np.multiply(meant,25);
        mean[t] = np.sum(meant)
      plt.ylabel('Sea ice extent (km\^2)'); plt.title('Sea ice extent [sum(area[sic>.15])]')
      figname=path_fig+run+'/sie_'+str(start_year)+'-'+str(start_month)+'_'+str(end_year)+'-'+str(end_month)+'.png'
  
    #time_series(time, sit_output, mask, 'test', 'Sea ice thickness time serie')
    time = timec #datac.time.indexes['time']
    plt.plot(time, mean, colors[ke-1])   
    #plt.xlabel('Time'); 
    #ll = [runs[i] for i in expt]
    for i in expt:
      ll.append(runs[i])

    plt.legend(ll)
    date_form = dates.DateFormatter("%b/%y")
    ax.xaxis.set_major_formatter(date_form)
    plt.tight_layout()
    if save_fig==1:
      if os.path.exists(path_fig+run)==False:
        os.mkdir(path_fig+run)
      plt.savefig(figname)
    plt.show()
  
  ### Plot video 
  if plot_video==1:
    plt.rcParams.update({'font.size': 12})
    if ke==1 and vname=='sie': # if first expt load obs
      fig, ax = plt.subplots(1,len(obs_sources)+len(expt)+len(expt), figsize = (16,8)) # landscape
      # plot obs
      if vname=='sie': # sea ice extent
        # loop in time to read obs
        kc=0; obs_colors=['g','y','orange']; ll=[]
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

          print('Processing SIC to get extent')
          sic_obs = np.zeros([np.shape(sicc_obs)[0],np.shape(lon_mod)[0],np.shape(lon_mod)[1]])
          for t in range(np.shape(sicc_obs)[0]):
            print('Processing SIC to get extent time: '+time_obs[t].strftime("%Y%m%d%HH:%MM"))
            #mean[t] = np.sum(sicc_obs[t]*25*25)
            sicct=sicc_obs[t]; 
            siccz=np.zeros((np.shape(sicct)[0],np.shape(sicct)[1])) 
            #iext=np.where(sicct>1); sicct[iext]=0;
            #iext=np.where(sicct>.15)[0]; sicct[iext]=1;
            iext=np.where(sicct>.15); 
            for ii in range(np.shape(iext)[1]):
              siccz[iext[0][ii],iext[1][ii]]=1.
            st = tictoc.time();   print('Interping obs to model grid ...'); # get the start time
            sicobsi=seapy.oasurf(np.array(lon_obs),np.array(lat_obs),np.array(siccz),np.array(lon_mod),np.array(lat_mod))[0]
            et = tictoc.time()-st; print('Execution time:', et, 'seconds')
            sicobsi[inan_mod]=np.nan
            sic_obs[t]=sicobsi

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
        if varim=='sic':
          cmap = cmocean.cm.ice
        elif varim=='sit':
          cmap = cmocean.cm.dense_r
        im1 = ax[ke-1].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmax = 1.0)
        fig.colorbar(im1, cax=cax, orientation='vertical')
  
        #def animate(i):
        #    im1.set_array(variable[i])
        #    ax[ke-1].set_title('Date :{} '.format(time[i].strftime('%Y.%m.%d')), loc = 'left')
        #    return [im1]
  
        #Nt = np.shape(variable)[0]
        #anim = animation.FuncAnimation(fig, animate, frames=len(time),
        #                               interval=interval, blit=True)

    if ke>=1 and vname=='sie': # if first expt load obs
      sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
      sic_mod = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
      sicc_mo=np.zeros((len(time_mod),np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
      for t in range(len(time_mod)): # (np.shape(sicc_mod)[0]):
        print('Processing model SIC to get extent time: '+time_mods[t].strftime("%Y%m%d%HH:%MM"))
        sicct=sic_mod[t];
        iext=np.where(sicct>.15)#[0]; sicct[iext]=1;
        for ii in range(np.shape(iext)[1]):
          sicct[iext[0][ii],iext[1][ii]]=1.
        sicc_mo[t]=sicct
      # daily average
      sicc_mod=np.zeros((len(time_obs),np.shape(sic_mod)[1],np.shape(sic_mod)[2]))
      for t in range(len(time_obs)): # (np.shape(sicc_mod)[0]):
        iday=np.where(time_obsn[t]==time_modi)[0]
        sicc_mod[t]=np.nanmean(sicc_mo[iday,:,:],axis=0)

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
      im2 = ax[ke].imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmax = 1.0)
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

    elif varim=='sic' or varim=='sit':

      plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
      sit_output = datac.sit.to_masked_array() # Extract a given variable
      time = datac.time.indexes['time']
  
      time=time_mod; mask =1- mask; 
      variable = sit_output;
      interval=10 #len(time))
      #anim=make_animation(time=time , mask =1- mask, variable = sit_output,interval=10)#len(time))
      #def make_animation(time,mask,variable,interval=10):
  
      fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))
      ax1.set_title('neXtSIM',loc='right')
      ax1.set_title('Date : {} '.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
      ax1.set_facecolor('xkcd:putty')
  
      # including colorbar
      divider = make_axes_locatable(ax1)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      if varim=='sic':
        cmap = cmocean.cm.ice
      elif varim=='sit':
        cmap = cmocean.cm.dense_r
      #im1=plt.pcolormesh(variable[0],cmap=cmap,animated=True,vmax = 2.5); 
      #plt.colorbar(); 
      im1 = ax1.imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmax = 3.5)
      #plt.colorbar()
      fig.colorbar(im1, cax=cax, orientation='vertical')
      #exit()
  
      def animate(i):
          im1.set_array(variable[i])
          ax1.set_title('Date :{} '.format(time[i].strftime('%Y.%m.%d')), loc = 'left')
          return [im1]
  
      Nt = np.shape(variable)[0]
      anim = animation.FuncAnimation(fig, animate, frames=len(time),
                                     interval=interval, blit=True)
      #plt.show()
      #return anim

  #exit()
  def animates(i):
    im1.set_array(sicc_obs[i])
    ax[0].set_title('Date :{} '.format(time_obs[i].strftime('%Y.%m.%d')), loc = 'left')
    im2.set_array(sicc_mod[i])
    im3.set_array(sicc_diff[i])
    return [im1,im2,im3]

  time=time_obs
  Nt = np.shape(variable)[0]
  anim = animation.FuncAnimation(fig, animates, frames=len(time),
                                     interval=interval, blit=True)
  FFwriter = animation.FFMpegWriter( fps = 24)
  ##Save animation 
  figname=path_fig+run+'/'+vname+'_'+str(start_year)+'-'+str(start_month)+'-'+str(start_day)+'_'+str(end_year)+'-'+str(end_month)+'-'+str(end_day)+'.mp4'
  if save_fig==1:
    print('Saving: '+figname)
    if os.path.exists(path_fig+run)==False:
      os.mkdir(path_fig+run)
    anim.save(figname, writer=FFwriter, dpi = 150)


  ### Make animation of sea-ice thickness
  if plot_anim==1:
  
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    sit_output = datac.sit.to_masked_array() # Extract a given variable
    time = datac.time.indexes['time']
  
    time=time; mask =1- mask; variable = sit_output;
    anim=make_animation_util(time=time , mask =1- mask, variable = sit_output,interval=10)#len(time))
  
    FFwriter = animation.FFMpegWriter( fps = 24)
    ##Save animation 
    figname=path_fig+run+'/sit_'+start_year+'-'+start_month+'_'+end_year+'-'+end_month+'.mp4'
    if save_fig==1:
      if os.path.exists(path_fig+run)==False:
        os.mkdir(path_fig+run)
      anim.save(figname, writer=FFwriter, dpi = 150)
  
  
  #fig=plt.subplots(); plt.pcolormesh(datac.sit[0,:,:]); plt.colorbar(); plt.title(time[0]); plt.show()
