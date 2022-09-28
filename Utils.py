import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import cmocean

from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc

def make_animation(time,mask,variable,
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


    
