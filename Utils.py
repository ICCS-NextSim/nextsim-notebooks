import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean
import cartopy
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from matplotlib import dates
import pandas as pd
import datetime

def daily_clim(time_obsd,mean):

  # climatological time
  time_ini = dates.date2num(datetime.datetime(2015,1,1,3,0,0))
  time_fin = dates.date2num(datetime.datetime(2015,12,31,3,0,0))
  freqobs  = 1; # daily data
  times=pd.date_range(dates.num2date(time_ini), periods=int(time_fin-time_ini)*freqobs, freq=('%dD' % int(1/freqobs)))
  time_clin=dates.date2num(times)
  time_cli=dates.num2date(time_clin)
  time_clid=pd.DatetimeIndex(time_cli)

  mean_cli=[]; std_cli=[]
  for t in range(len(time_clid)): # (np.shape(sicc_mod)[0]):
    m=time_clid[t].month; d=time_clid[t].day
    if d==29 and m==2:
      print('NOT COMPUTING FOR 29/2 '+str(d)+'/'+str(m).zfill(2))
    else:
      #print('computing daily longterm mean for '+str(d)+'/'+str(m).zfill(2))
      iday=time_obsd.day==d
      imonth=time_obsd.month==m; iym=np.where(iday*imonth==True)
      #time_cli.append(time[iym[0][0]])
      mean_cli.append(np.nanmean(mean[iday*imonth],axis=0)) # month average
      std_cli.append(np.nanstd(mean[iday*imonth],axis=0)) # month average

  mean_cli=np.array(mean_cli)  
  std_cli=np.array(std_cli)  

  return time_cli, mean_cli, std_cli

def veccor1(u1,v1,u2,v2):
    ''' 
    # [a,theta]=VECCOR1(u1,v1,u2,v2) computes the complex vector correlation
    # coefficient following Kundu (1976), JPO, 6, 238-242. Input are the
    # four time series vectors. Output is complex, with amplitude and
    # rotation angle in degrees. A positive angle indicates that series 1
    # is rotated positively (counterclockwise) from series 2.
    #
    ########################################################################
    # ver. 1: 12/1/96 (R.Beardsley)
    # ver. 2: allow for complex arguments, remove mean
    # brought (as is) to atsee collection in August 1999 by JiM.
    # Adpated from matlab to python by Rafa Santana, 30/01/2023
    ''' 
    import numpy as np
    from sys import exit
  
    # converting the vectors to complex numbers
    #  X=u1(:)+i*v1;
    #u1=u1.flatten(); v1=v1.flatten()
    #u2=u2.flatten(); v2=v2.flatten()
    X=np.transpose([u1.flatten()]) + 1j*v1.flatten()
    Y=np.transpose([u2.flatten()]) + 1j*v2.flatten()

    X=np.complex64(X)
    Y=np.complex64(Y)

    ## work on the common good points only
    #ii=find((isfinite(X+Y)));
    ii=np.isfinite(X+Y)==1;

    ## if no common good points, return NaNs
    if(np.sum(ii)<1):
      ac=np.NaN; theta=np.NaN; trans=np.NaN;
      return ac,theta 
      exit()

    X=X[ii]; 
    # there is seem to be some memory problem, in which "Y" gets 0 values
    # computing "Y" again solves the problem
    #Y=np.transpose([uc_mod[0,::v_spave,::v_spave].flatten()]) + 1j*vc_mod[0,::v_spave,::v_spave].flatten()
    Y=np.transpose([u2.flatten()]) + 1j*v2.flatten()
    Y=np.complex64(Y)
    Y=Y[ii];
    # if that doesnt work swap the input (e.g. model vs obs or obs vs model)
 
    ## remove mean
    X=X-np.nanmean(X[:]); 
    Y=Y-np.nanmean(Y[:]);
    ## compute a, theta
    #c=(rot90(X)*conj(Y))./(sqrt(rot90(X)*conj(X))*sqrt(rot90(Y)*conj(Y)));
    #c = np.rot90([X],k=1,axes=(0,1)) * np.conj(Y)  / (np.sqrt(np.rot90([X],k=1,axes=(0,1))*np.conj(X))*np.sqrt(np.rot90([Y],k=1,axes=(0,1))*np.conj(Y)));
    numera = np.matmul(np.rot90([X],k=1,axes=(1,0)).T , np.conj(Y)) 
    denomi = (np.sqrt ( np.matmul(np.rot90([X],k=1,axes=(1,0)).T , np.conj(X)) ) * np.sqrt( np.matmul(np.rot90([Y],k=1,axes=(1,0)).T , np.conj(Y)) ) );

    c=np.divide(numera,denomi);
    ac=np.abs(c);
    theta=180*np.angle(c)/np.pi;
    
    return ac,theta,X,Y 

def text_map_w_stats(ax,data,lon_mod,bm,lon_regions,lat_regions,latn,oper,unit,colort):
    ''' 
    text_map_w_stats(ax,data,lon_mod,bm,lon_regions,lat_regions,latn,oper,unit,colort):
    ax = fig ax
    data = input 2D data
    lon_mod = model longitude
    bm = Basemap() output
    lon_regions = longitudinal sections to compute stats. starting from the smallest e.g. -179
    lat_regions = latitudinal sections to compute stats
    latn = latitude that receives text
    oper = operation, e.g.: sum, mean, ... if '' (no operation nor texting)
    unit = m, m3, etc
    colort = color of text

    This function computes and writes stats in the regions between
    two lon_regions longitudes. Operations available are sum and mean
     
    Rafa Santana, 14/04/2023
    ######################################################################
    ''' 
    import numpy as np
    from sys import exit

    lon_regions=np.array(lon_regions);
    lat_regions=np.array(lat_regions);
    rnames=np.array(['Ross','AB Seas','Weddell','Atlantic','Indian','Pacific']);
    lon_r360=np.where(lon_regions>=0,lon_regions,lon_regions+360); # lon 0-360

    for kl in range(0,len(lon_regions)):
      lsec=lon_regions[kl]
      lasec=lat_regions[kl]

      if lsec==lon_regions[0]: # values crossing the 180E/W line
        dataf=np.where(lon_mod<lon_regions[0],data,0); dataf2=np.where(lon_mod>lon_regions[-1],data,0); dataf=dataf+dataf2
        dataf=np.where(dataf!=0,dataf,np.nan)
      else: # other regions
        dataf=np.where(lon_mod>lon_regions[kl-1],data,np.nan); dataf=np.where(lon_mod<lon_regions[kl],dataf,np.nan)

      if oper=='sum':
        dataf=format(np.nansum(dataf),'.2f')
      elif oper=='mean':
        dataf=format(np.nanmean(dataf),'.2f')

      if kl==0: # values crossing the 180E/W line
        lon_r=lon_r360
      else:
        lon_r=lon_regions

      # texting sections
      if oper=='':
        x,y = bm(np.nanmean([lon_r[kl-1],lon_r[kl]])-0,latn);
        ax.annotate(rnames[kl], xy=(x, y), xycoords='data', xytext=(x, y),fontsize=9,color=colort,fontweight='bold')#, textcoords='offset points',
      else:
        x,y = bm(np.nanmean([lon_r[kl-1],lon_r[kl]])-0,latn);
        ax.annotate(dataf+' '+unit, xy=(x, y), xycoords='data', xytext=(x, y),fontsize=9,color=colort,fontweight='bold')#, textcoords='offset points',
        #color='r', arrowprops=dict(arrowstyle="->")) #"fancy", color='g')

      # plotting lines
      x,y = bm([lsec,lsec],[lasec, -50]); bm.plot(x,y,color=colort,linewidth=1,)

def format_map(ax):
    ax.gridlines(zorder=2,linewidth=0.25,linestyle="--",color="darkgrey")
    ax.set_extent([-360, 180, -54, -90], ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND,zorder=1,alpha=0.7,facecolor="lightgreen")


def make_animation_util(time,mask,variable,
                           interval=10):
    '''
    Animation of nextsim simulation outputs for one variable
    time : time serie
    mask : mask of the dataset
    variable : variable of interest
    interval : time (ms) between 2 frames (typically 20ms)
     '''
    
    fig, ax1 = plt.subplots(1, 1 ,figsize=(8,8))  
    ax1.set_title('neXtSIM',loc='right')
    ax1.set_title('Date : {} '.format(time[0].strftime('%Y.%m.%d')), loc = 'left')
    ax1.set_facecolor('xkcd:putty')
    
    cmap = cmocean.cm.ice
    im1 = ax1.imshow(variable[0],cmap=cmap,origin = 'lower',animated=True,vmax = 2.5)

    def animate(i):
        im1.set_array(variable[i])
        ax1.set_title('Date :{} '.format(time[i].strftime('%Y.%m.%d')), loc = 'left')
        return [im1]
    Nt = np.shape(variable)[0]

    anim = animation.FuncAnimation(fig, animate, frames=len(time),
                                   interval=interval, blit=True)
    #plt.show()
    return anim

def time_series(time, variable, mask, save_name, title): 
    ''' 
    plot the time serie mean of any Masked Array and save the figure
    Inputs : 
    - time (datetime array)
    - vzriable : variable of interest
    - mask : mask of the variable
    - save_name : figure name for saving
    - title : plot title
    '''
    
    T = np.shape(variable)[0]
    mean = np.zeros(T)
    std = np.zeros(T)
    for t in range(T):
        mean[t] = np.mean(variable[t][variable[t].mask == False])
    plt.plot(time, mean, '.--b')

def time_series_plot(time, variable, mask, save_name, title): 
    ''' 
    plot the time serie mean of any Masked Array and save the figure
    Inputs : 
    - time (datetime array)
    - vzriable : variable of interest
    - mask : mask of the variable
    - save_name : figure name for saving
    - title : plot title
    '''
    
    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    T = np.shape(variable)[0]
    mean = np.zeros(T)
    std = np.zeros(T)
    for t in range(T):
        mean[t] = np.mean(variable[t][variable[t].mask == False])
    plt.plot(time, mean, '.--')
    plt.xlabel('Time')
    plt.ylabel('Domain average SIT (m)')
    plt.title(title)
    plt.savefig(save_name +'.png')
    
def time_series_plot2(time, variable, time2, variable2, time3, variable3, mask, save_name, title): 
    ''' 
    plot the time serie mean of any Masked Array and save the figure
    Inputs : 
    - time (datetime array)
    - vzriable : variable of interest
    - mask : mask of the variable
    - save_name : figure name for saving
    - title : plot title
    '''
    
    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    T = np.shape(variable)[0]; T2 = np.shape(variable2)[0]; T3 = np.shape(variable3)[0]
    mean = np.zeros(T); mean2 = np.zeros(T2);  mean3 = np.zeros(T3)
    
    for t in range(T):
        mean[t] = np.mean(variable[t][variable[t].mask == False])
    for t in range(T2):
        mean2[t] = np.mean(variable2[t][variable2[t].mask == False])
    for t in range(T3):
        mean3[t] = np.mean(variable3[t][variable3[t].mask == False])
        
    plt.plot(time, mean, '.--',label='ocean-only')
    plt.plot(time2, mean2, '.--',label='ocean+thermodynamics')
    #plt.plot(time3, mean3, '.--',label='ocean+wind+thermodynamics')
    plt.xlabel('Time')
    plt.legend()                      
    plt.ylabel('Domain average SIT (m)')
    plt.title(title)
    plt.savefig(save_name +'.png')

    
def time_series_plot3(time, variable, time2, variable2, time3, variable3, mask, save_name, title): 
    ''' 
    plot the time serie mean of any Masked Array and save the figure
    Inputs : 
    - time (datetime array)
    - vzriable : variable of interest
    - mask : mask of the variable
    - save_name : figure name for saving
    - title : plot title
    '''
    
    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    T = np.shape(variable)[0]; T2 = np.shape(variable2)[0]; T3 = np.shape(variable3)[0]
    mean = np.zeros(T); mean2 = np.zeros(T2);  mean3 = np.zeros(T3)
    
    for t in range(T):
        mean[t] = np.mean(variable[t][variable[t].mask == False])
    for t in range(T2):
        mean2[t] = np.mean(variable2[t][variable2[t].mask == False])
    for t in range(T3):
        mean3[t] = np.mean(variable3[t][variable3[t].mask == False])
        
    plt.plot(time, mean, '.--',label='ocean-only')
    plt.plot(time2, mean2, '.--',label='ocean+thermodynamics')
    plt.plot(time3, mean3, '.--',label='ocean+wind+thermodynamics')
    plt.xlabel('Time')
    plt.legend()                      
    plt.ylabel('Domain average SIT (m)')
    plt.title(title)
    plt.savefig(save_name +'.png')


    
