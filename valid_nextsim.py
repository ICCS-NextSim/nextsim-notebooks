import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from Utils import * #make_animation, time_series_plot, time_series_plot2
from sys import exit
plt.ion()
plt.close('all')

# Plot types
plot_series=1
plot_map   =1

#Variables
vari='sit'

#Runs
run='data_southern'

#Time
start_month=1
start_year =2019
end_month  =12
end_year   =2019
#trick to cover all months in runs longer than a year
end_month=end_month+1
ym_start= 12*start_year + start_month - 1
ym_end  = 12*end_year + end_month - 1

#Colors


####################################################################
#paths
path_runs='~/'
path_fig='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/figures/'

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
plt.plot(time, mean, 'b')    
time = datac.time.indexes['time']
plt.xlabel('Time')
plt.ylabel('SIT (m)')
plt.title('Domain average sea ice thickness (SIT)')
figname=path_fig+run+'/domain_average_sit_'+str(start_year)+'-'+str(start_month)+'_'+str(end_year)+'-'+str(end_month)+'.png'
plt.savefig(figname)
plt.show()


### Make animation of sea-ice thickness
if plot_map==1:



  plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
  sit_output = datac.sit.to_masked_array() # Extract a given variable
  time = datac.time.indexes['time']
  anim=make_animation(time=time , mask =1- mask, variable = sit_output,interval=10)#len(time))
  FFwriter = animation.FFMpegWriter( fps = 24)
  ##Save animation 
  anim.save(path_fig+run+'/sic_'+start_year+'-'+start_month+'_'+end_year+'-'end_month'.mp4', writer=FFwriter, dpi = 150)


