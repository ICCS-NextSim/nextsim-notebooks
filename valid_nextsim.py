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
plot_series=0
plot_map   =1

#Runs

#Time
start_month=1
start_year =2018
end_month  =3
end_year   =2018
#trick to cover all months in runs longer than a year
end_month=end_month+1
ym_start= 12*start_year + start_month - 1
ym_end  = 12*end_year + end_month - 1

#Colors


#Grid information
data = xr.open_dataset('~/data/output/Moorings_2018m01.nc')
sit_output = data.sit.to_masked_array() # Extract a given variable
mask = ma.getmaskarray(sit_output[0]) #Get mask

if plot_series==1:
  # Plotting time series
  fig, ax = plt.subplots(1, 1, figsize = (15,5))
  #for month in ['01','02','03']:
  for ym in range( ym_start, ym_end ):
    y, m = divmod( ym, 12 ); m+=1
    if m<10:
      filename='~/data/output/Moorings_'+str(y)+'m0'+str(m)+'.nc'
    else:
      filename='~/data/output/Moorings_'+str(y)+'m'+str(m)+'.nc'
    print(filename)
    data = xr.open_dataset(filename)
    if m==1:
      datac = data
    else:
      datac = xr.concat([datac,data],'time')

  sit_output = datac.sit.to_masked_array() # Extract a given variable
  time = datac.time.indexes['time']
  time_series(time, sit_output, mask, 'test', 'Sea ice thickness time serie')
  plt.xlabel('Time')
  plt.ylabel('SIT (m)')
  plt.title('Domain average sea ice thickness (SIT)')
  plt.savefig('test'+'.png')
  plt.show()


### Make animation of sea-ice thickness
if plot_map==1:

  plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
  for ym in range( ym_start, ym_end ):
    y, m = divmod( ym, 12 ); m+=1
    if m<10:
      filename='~/data/output/Moorings_'+str(y)+'m0'+str(m)+'.nc'
    else:
      filename='~/data/output/Moorings_'+str(y)+'m'+str(m)+'.nc'
    print(filename)
    data = xr.open_dataset(filename)
    if m==1:
      datac = data
    else:
      datac = xr.concat([datac,data],'time')

  sit_output = datac.sit.to_masked_array() # Extract a given variable
  time = datac.time.indexes['time']
  anim=make_animation(time=time , mask =1- mask, variable = sit_output,interval=10)#len(time))
  FFwriter = animation.FFMpegWriter( fps = 24)
  ##Save animation 
  anim.save('sit_all_2018-01_2018-03.mp4', writer=FFwriter, dpi = 150)


