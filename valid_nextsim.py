import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from matplotlib import dates
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utils import * #make_animation, time_series_plot, time_series_plot2
from sys import exit
import os
import socket
plt.ion()
plt.close('all')

#Time
start_day  =1
start_month=1
start_year =2018
end_day    =30
end_month  =8
end_year   =2019
#end_month  =1
#end_year   =2018

#Runs (names) or experiments (numbers)
expt=[0,1]

# Plot types
plot_series=1
plot_map   =0
plot_video =0
plot_anim  =0
save_fig   =0

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
vname='sic' # timeseires
varim='sit' # video

#trick to cover all months in runs longer than a year
end_month=end_month+1
ym_start= 12*start_year + start_month - 1
ym_end  = 12*end_year + end_month - 1
end_month=end_month-1

#paths
 
if socket.gethostname()=='SC442555':
  path_runs='/Users/rsan613/n/southern/runs/' # ''~/'
  path_fig ='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/figures/'
  path_data ='/Users/rsan613/n/southern/data/'
else:
  print("Your runs, figures etc paths haven't been set")
  exit()
  
#Grid information
run=runs[expts[0]] # 'data_glorys'
data = xr.open_dataset(path_runs+run+'/output/Moorings_2018m01.nc')
sit_output = data.sit.to_masked_array() # Extract a given variable
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
      vdatac = data.variables[vname]
    else:
      #datac = xr.concat([datac,data],'time')
      time = data.variables['time'];   timec = xr.Variable.concat([timec,time],'time')
      sic = data.variables['sic'];   sicc = xr.Variable.concat([sicc,sic],'time')
      vdata = data.variables[vname]; vdatac = xr.Variable.concat([vdatac,vdata],'time')
      #exit() 
    data.close()

  #datac.data_vars
  
  if plot_series==1:
    plt.rcParams.update({'font.size': 22})
    # Plotting time series
    if ke==1:
      fig, ax = plt.subplots(1, 1, figsize = (16,8)) # landscape

      # plot obs
      if vname=='sic': # sea ice extent
        # loop in time to read obs
        kc=0; obs_colors=['g','y','orange']; ll=[]
        for obs_source in ['OSISAF-ease']: #['NSIDC','OSISAF','OSISAF-ease']:
          ll.append(['OBS-'+obs_source]); k=0; kc+=1
          if obs_source[0:11]=='OSISAF-ease':
            file=path_data+'/sic_osisaf/2018'+'/ice_conc_sh_ease-125_multi_20180101'+'.nc';
            data = xr.open_dataset(file)
            xobs = data.variables['xc']; yobs = data.variables['yc']
            dx,dy=np.meshgrid(np.diff(xobs),np.diff(yobs)); dy=np.abs(dy); obs_grid_area=dx*dy
          for t in time_obs:
            k+=1
            if obs_source=='NSIDC':
              file=path_data+'/sic_nsidc/'+t.strftime("%Y")+'/'+prefix_sic+t.strftime("%Y%m%d")+sufix_sic
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


          mean = np.zeros(np.shape(sicc_obs)[0])
          for t in range(np.shape(sicc_obs)[0]):
            #mean[t] = np.sum(sicc_obs[t]*25*25)
            sicct=sicc_obs[t]; 
            #exit() 
            iext=np.where(sicct>1)[0]; sicct[iext]=0;
            #iext=np.where(sicct>=.15)[0]; sicct[iext]=1;
            #iext=np.where(sicct<.15)[0]; sicct[iext]=0;
            #mean[t] = np.sum(sicct*25*25)
            if obs_source[0:11]=='OSISAF-ease':
              meant = np.multiply(sicct[0:-1,0:-1],obs_grid_area); # meant = np.multiply(meant,obs_grid_area);
            else:
              meant = np.multiply(sicct,obs_grid_area); meant = np.multiply(meant,obs_grid_area);
            mean[t] = np.sum(meant)


          plt.plot(time_obs, mean, color=obs_colors[kc-1])   
          plt.grid()
          #exit()
          #expt=np.sum(expt,1)
 
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
    elif vname=='sic':
      sit = vdatac;  #_output = datac.sit.to_masked_array() # Extract a given variable
      sic = sicc #_output = datac.sit.to_masked_array() # Extract a given variable
      T = np.shape(sit)[0]
      mean = np.zeros(T)
      std = np.zeros(T)
      for t in range(T):
        #mean[t] = np.sum(sic[t]*50*50)
        sicct=sic[t];
        iext=np.where(sicct>1)[0]; sicct[iext]=0;
        #iext=np.where(sicct>.15)[0]; sicct[iext]=1;
        #iext=np.where(sicct<.15)[0]; sicct[iext]=0;
        meant = np.multiply(sicct,25); meant = np.multiply(meant,25);
        mean[t] = np.sum(meant)
      plt.ylabel('Sea ice total area (km\^2)'); plt.title('Sea ice total area (sum of [grid area * SIC])')
      figname=path_fig+run+'/domain_average_sit_'+str(start_year)+'-'+str(start_month)+'_'+str(end_year)+'-'+str(end_month)+'.png'
  
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
  ### Make animation of sea-ice thickness
  if plot_video==1:
  
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    sit_output = datac.sit.to_masked_array() # Extract a given variable
    time = datac.time.indexes['time']
  
    time=time; mask =1- mask; 
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
  
    FFwriter = animation.FFMpegWriter( fps = 24)
    ##Save animation 
    figname=path_fig+run+'/sit_'+str(start_year)+'-'+str(start_month)+'_'+str(end_year)+'-'+str(end_month)+'.mp4'
    if save_fig==1:
      if os.path.exists(path_fig+run)==False:
        os.mkdir(path_fig+run)
      anim.save(figname, writer=FFwriter, dpi = 150)


