import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Utils import * #make_animation, time_series_plot, time_series_plot2
from sys import exit
plt.ion()
plt.close('all')

#Time
start_month=1
start_year =2019
end_month  =12
end_year   =2019

#Runs
runs=[0,1]

run='data_southern'

# Plot types
plot_series=1
plot_map   =0
plot_anim  =0

#Colors
colors={'r','b','k'}

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
expts={'data_glorys','data'}


#Variables
varim='sit'
vname='mean_sit'


#trick to cover all months in runs longer than a year
end_month=end_month+1
ym_start= 12*start_year + start_month - 1
ym_end  = 12*end_year + end_month - 1
end_month=end_month-1

#paths
path_runs='~/'
path_fig ='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/figures/'

#Grid information
data = xr.open_dataset(path_runs+run+'/output/Moorings_2018m01.nc')
sit_output = data.sit.to_masked_array() # Extract a given variable
mask = ma.getmaskarray(sit_output[0]) #Get mask

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
    datac = data
  else:
    datac = xr.concat([datac,data],'time')

datac.data_vars

if plot_series==1:
  plt.rcParams.update({'font.size': 22})
  # Plotting time series
  fig, ax = plt.subplots(1, 1, figsize = (15,5))

  if  vname=='mean_sit':
    #sit_output = datac.sit.to_masked_array() # Extract a given variable
    sit = datac.sit.to_masked_array() # Extract a given variable
    sic = datac.sic.to_masked_array() # Extract a given variable
    T = np.shape(sit)[0]
    mean = np.zeros(T)
    std = np.zeros(T)
    for t in range(T):
        #mean[t] = np.mean(variable[t][variable[t].mask == False])
        mean[t] = np.mean((sit[t]*sic[t])/sic[t])

  #time_series(time, sit_output, mask, 'test', 'Sea ice thickness time serie')
  time = datac.time.indexes['time']
  plt.plot(time, mean, 'b')    
  plt.xlabel('Time')
  plt.ylabel('SIT (m)')
  plt.title('Domain average sea ice thickness (SIT)')
  figname=path_fig+run+'/domain_average_sit_'+str(start_year)+'-'+str(start_month)+'_'+str(end_year)+'-'+str(end_month)+'.png'
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
  anim.save(figname, writer=FFwriter, dpi = 150)


#fig=plt.subplots(); plt.pcolormesh(datac.sit[0,:,:]); plt.colorbar(); plt.title(time[0]); plt.show()
### Make animation of sea-ice thickness
if plot_map==1:

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
  anim.save(figname, writer=FFwriter, dpi = 150)


