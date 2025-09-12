global startdate; startdate=datenum(2017,02,01,00,00,00);
global dt; dt=60;
global namestep;namestep=3600/dt;
global nx; nx=210;
global ny; ny=240;
global nz; nz=70;


%% grid - centers of cells

XC=rdmds('XC');% xaxis
YC=rdmds('YC');% yaxis
RC=rdmds('RC');% zaxis

%% land mask 
hfac=rdmds('hFacC');
mask(:,:)=hfac(:,:,1);
nanmask=mask;
nanmask(mask==0)=NaN;



%% read timeseries 

for time=1:24*(333)
    time_mit=60 + (time-1)*60;
    plotter_time_mit(time)= startdate + ( time_mit)*dt/(3600*24) ;   %time variable
      
    K=rdmds('stateICE',time_mit);
    area(:,:,time)=K(:,:,1);    %sea ice concentration
    heff(:,:,time)=K(:,:,2);    % effective sea ice thickness
    hsnow(:,:,time)=K(:,:,3);    % effective see ice snow
end
%%

datestr(plotter_time_mit(1))

datestr(plotter_time_mit(end))
%%

% ice thickenss = heff / area
% snow thickness = hsnow / area


