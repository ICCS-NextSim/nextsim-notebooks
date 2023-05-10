clear
close all  

%Time
start_day  =1;
start_month=1;
start_year =2016;
end_day    =31;
end_month  =12;
end_year   =2017;


%Runs (names) or experiments (numbers - starts with 1)
expt=[12,9,17,15];%2,5,7,10]
expt=[17,9];

serie_or_maps=[0]; % 1 for serie, 2 for video, 3 for map, 0 for neither
my_dates=1;
inc_obs=1;

% Plot types
plot_scatter=0;
plot_series =0;
plot_video  =0 ; 
plot_map    =1;
plot_anim   =0;
save_fig    =0;
plt_show    =1;
interp_obs  =1 ;% only for SIE maps obs has 2x the model resolution

%Variables
vname ='newice_perc'; % newice_perc 
varim =''; % 'sit' for model solo videos  % video

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% after 'BSOSE' run (ocean boundary cond), runs are all mEVP
runs={'50km_ocean_wind'     ,'50km_bsose_20180102'   ,'50km_hSnowAlb_20180102','50km_61IceAlb_20180102','50km_14kPmax_20180102',...   % 5
      '50km_20Clab_20180102','50km_P14C20_20180102'  ,'50km_LandNeg2_20180102','50km_bsose_20130102'   ,'50km_dragWat01_20180102',... % 10
      '50km_glorys_20180102','BSOSE'                 ,'50km_mevp_20130102'    ,'50km_lemieux_20130102' ,'50km_h50_20130102',...       % 15
      '50km_hyle_20130102'  ,'50km_ckFFalse_20130102'}; % ,'50km_mevp_20130102'    ,'50km_lemieux_20130102' ,'50km_h50_20130102']

expts=1:length(runs); %) #[0,1,2,3,4,5]

%Colors
colors={'r','b','k','r','m','b','y','g','r','b','k'};
obs_colors={'g','y','r'};

% varrays according to vname
if vname=='newice_perc' 
  varray='newice'; 
end

%trick to cover all months in runs longer than a year
end_month=end_month+1;
ym_start= 12*start_year + start_month - 1;
ym_end  = 12*end_year + end_month - 1;
end_month=end_month-1;


% SIE obs sources
obs_sources={'OSISAFease2'};%,'OSISAF-ease'] %['NSIDC','OSISAF','OSISAF-ease','OSISAFease2']: 

%paths
%print('Hostname: '+socket.gethostname())
%if socket.gethostname()=='SC442555' or socket.gethostname()=='SC442555.local' or socket.gethostname()=='wifi-staff-172-24-40-164.net.auckland.ac.nz':
%  path_runs='/Users/rsan613/n/southern/runs/' % ''~/'
%  path_fig ='/Users/rsan613/Library/CloudStorage/OneDrive-TheUniversityofAuckland/001_WORK/nextsim/southern/figures/'
%  path_data ='/Users/rsan613/n/southern/data/'
%  path_bsose='/Volumes/LaCie/mahuika/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/bsose/'
%elif socket.gethostname()[0:3]=='wmc' or socket.gethostname()=='mahuika01' or socket.gethostname()=='mahuika':
  path_runs='/scale_wlg_persistent/filesets/project/uoa03669/rsan613/n/southern/runs/'; % ''~/'
  path_fig ='/scale_wlg_persistent/filesets/project/uoa03669/rsan613/n/southern/figures/'; 
  path_data ='/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/';
  path_bsose='/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/bsose/';
%else:
%  display('Your data, runs and figures paths havent been set')
%  exit()
  
%Grid information
run=runs{8}; %expt[0]] % 'data_glorys'


ncfile = strcat(path_runs,run,'/output/Moorings_2018m01.nc');
lon_mod = ncread(ncfile,'longitude'); %sit.to_masked_array() % Extract a given variable
lat_mod = ncread(ncfile,'latitude'); %sit.to_masked_array() % Extract a given variable
%lon_mod=np.where(lon_mod!=np.max(lon_mod),lon_mod,179.99999999999)%180.01)
%lon_mod=np.where(lon_mod!=np.min(lon_mod),lon_mod,-179.99999999999)%-180.01)
lon_nex = lon_mod; 
lat_nex = lat_mod; 
v_spam=10;
lon_modv=lon_mod(1:v_spam:end,1:v_spam:end);
lat_modv=lat_mod(1:v_spam:end,1:v_spam:end);
%sit_output = data.sit.to_masked_array() % Extract a given variable
%inan_mod=ma.getmaskarray(sit_output[0]); 
%mask = ma.getmaskarray(sit_output[0]) %Get mask

% time_obs
time_ini = datenum(start_year,start_month,start_day);
time_fin = datenum(end_year,end_month,end_day);
freqobs  = 1; % daily data
times=time_ini:freqobs:time_fin; %pd.date_range(dates.num2date(time_ini), periods=int(time_fin-time_ini)*freqobs, freq=('%dD' % int(1/freqobs)))
time_obsn=times; % dates.date2num(times)
time_obs=time_obsn;
%time_obsd=pd.DatetimeIndex(time_obs)

%return

% Loop in the experiments
ke=0;
for ex = expt;
  ke=ke+1;
  run=runs{expts(ex)};

  % Loading data
  if strcmp(run,'BSOSE');
%    if varray=='sic'; %vname=='sie':
%      filename=path_bsose+'SeaIceArea_bsoseI139_2013to2021_5dy.nc'
%      print(filename)
%      ds=xr.open_dataset(filename)
%      sicc=ds.variables['SIarea'][:] 
%      sic_mod = sicc %_output = datac.sit.to_masked_array() % Extract a given variable
%      vdatac=sicc
%      %data = xr.open_dataset(filename)
%      %timec = data.variables['time']; sicc = data.variables['SIarea']; 
%    elseif varray=='sit'; %vname=='sie':
%      filename=path_bsose+'SeaIceArea_bsoseI139_2013to2021_5dy.nc'
%      print(filename)
%      ds=xr.open_dataset(filename)
%      sicc=ds.variables['SIarea'][:] 
%      sic_mod = sicc %_output = datac.sit.to_masked_array() % Extract a given variable
%      filename=path_bsose+'SeaIceHeff_bsoseI139_2013to2021_5dy.nc'
%      print(filename)
%      ds=xr.open_dataset(filename)
%      vdatac=ds.variables['SIheff'][:] 
%      %sic_mod = sicc %_output = datac.sit.to_masked_array() % Extract a given variable
%      %vdatac=sicc
%      %data = xr.open_dataset(filename)
%      %timec = data.variables['time']; sicc = data.variables['SIarea']; 
%    elseif varray=='siv':
%      filename=path_bsose+'SIuice_bsoseI139_2013to2021_5dy.nc'
%      print(filename)
%      ds=xr.open_dataset(filename)
%      udatac=ds.variables['SIuice'][:] 
%      filename=path_bsose+'SIvice_bsoseI139_2013to2021_5dy.nc'
%      print(filename)
%      ds=xr.open_dataset(filename)
%      vdatac=ds.variables['SIvice'][:] 
%    end
%    filename=path_bsose+'SeaIceArea_bsoseI139_2013to2021_5dy.nc'; ds=xr.open_dataset(filename)
%    lon_sose=ds.variables['XC'][:]
%    lon_sose=np.where(lon_sose<180,lon_sose,lon_sose-360)
%    lat_sose=ds.variables['YC'][:]
%    area_sose=ds.variables['rA'][:]/1000000.
%    lon_mod,lat_mod=np.meshgrid(lon_sose,lat_sose);
%    timec=ds.variables['time'][:]%/(3600*24) % making SOSE date centered as the 5-day average 
%    ds.close()
%    %time_date=dates.num2date(time_in,"seconds since 2012-12-01")
%    %time_out=date2num(time_date,"hours since 1950-01-01 00:00:00")
%    time_mod=dates.date2num(timec)
%    time_mods=dates.num2date(time_mod)
%    time_modd=pd.DatetimeIndex(time_mods)
%    time_modi=[int(time_mod[ii]) for ii in range(len(time_mod))] % integer time for daily search
%
  else;
    k=0;
    for ym=ym_start:ym_end-1
      k=k+1;
      y=fix(ym/12); 
      m=mod(ym,12); m=m+1;
      filename=strcat(path_runs,run,'/output/Moorings_',num2str(y),'m',num2str(m,'%02i'),'.nc');
      display(['Loading: ',filename])
      if k==1
        timec = ncread(filename,'time')+datenum(1900,01,01); 
        sicc = ncread(filename,'sic'); vdatac = ncread(filename,varray); %['sit']
        if strcmp(varray,'siv')
          udatac = ncread(filename,'siu');
        end
        v_spam=10;
        lon_modv=lon_mod(1:v_spam:end,1:v_spam:end);
        lat_modv=lat_mod(1:v_spam:end,1:v_spam:end);
        inan_mod=find(sicc==nan); % ma.getmaskarray(sit_output[0]); 
      else
        time = ncread(filename,'time')+datenum(1900,01,01); timec = [timec;time];
        sic = ncread(filename,'sic');   sicc = cat(3,sicc,sic);
        vdata = ncread(filename,varray); vdatac = cat(3,vdatac,vdata);
        if strcmp(varray,'siv')
          udata = ncread(filename,'siu'); 
          udatac = cat(3,udatac,udata);
        end %exit() 
      end
      %data.close()

      lon_mod = lon_nex; 
      lat_mod = lat_nex;
      time_mod=timec;
      time_mods=datevec(time_mod);
      %time_modd=pd.DatetimeIndex(time_mods)
      time_modi=fix(time_mod); % [int(time_mod[ii]) for ii in range(len(time_mod))] % integer time for daily search
      %exit()

    end % loop in time

  end % expt type: BSOSE or nextsim
%%
%%    %datac.data_vars
%  
  if plot_map==1;

		if strcmp(vname,'newice_perc' )
      
      filepxx=[path_runs,run,'/output/newice_pxx_',datestr(time_mod(1),'yyyy-mm-dd'),'_',datestr(time_mod(end),'yyyy-mm-dd'),'.mat'];
      if exist(filepxx)==2
        display(['Loading: ',filepxx])
        load([filepxx])
      else
			  %% power spectrum analysis
			  display(['Computing ','power spectrum analysis per gird point'])
			  for i = 1:size(lat_mod,1) 
			    display(['i = ',num2str(i),' of ',num2str(size(lat_mod,1))])
			  	for j = 1:size(lat_mod,2)
			  		fld = squeeze(vdatac(i,j,:)); 
			  		if sum(isnan(fld)) == 0
			  			[pxx(i,j,:),f] = pwelch(fld); 
			  		end
			  	end
			  end
        display(['Saving: ',filepxx])
        save([filepxx],'pxx','f')
      end
      if ke==1
       pxxst=pxx; 
      else
       pxx=pxx-pxxst; 
      end
			%% 
			% Original delta t = 6 hr, so that means delta f is cycles/6 hr. 
			% 1/f gives periods in 6hr/cylce. f/4 gives days/cycle
			% 2pif/4 gives days
			T = 2*pi./(4*f); % (T is period in  days from the longest to the shortest period (longest = 256 days if input is 2 years. 1/2 day = 6h delta t ))
			ishort = find(T < 2,1); 
			ilong = find(T(2:end)<60,1); 
      scrsz=[1 1 1366 768];
      scrsz=get(0,'screensize');

			%close all
      figure('position',scrsz,'color',[1 1 1],'visible','on');  
			ax=subplot(121);
      hold on; 
			worldmap([-90 -60],[-180 180])
			model=sum(pxx(:,:,ilong:ishort),3); 
			pcolorm(double(lat_mod),double(lon_mod),model); 
      colorbar
      if ke==1
			  set(gca,'clim',[0 .001])
			  colormap(ax,cmocean('delta')); 
        title('Energy in periods between 60 and 2 days')
      else
			  set(gca,'clim',[-.001 .001])
			  colormap(ax,cmocean('balance')); 
        title('Energy difference in periods between 60 and 2 days')
      end

			ax=subplot(122);
      hold on; 
			worldmap([-90 -60],[-180 180])
			model=sum(pxx(:,:,ishort:end),3); 
			pcolorm(double(lat_mod),double(lon_mod),model); 
      colorbar
      if ke==1
			  set(gca,'clim',[0 .001])
			  colormap(ax,cmocean('delta')); 
        title('Energy in periods shorter than 2 days')
      else
			  set(gca,'clim',[-.001 .001])
			  colormap(ax,cmocean('balance')); 
        title('Energy difference in periods shorter than 2 days')
      end

		end %  if strcmp(vname,'newice_perc')

  end % plot_map


end % loop in expts

%display('End of script')
%%plt.close('all')
