clear
close all  

%Time
start_day  =1;
start_month=2;
start_year =2017;

end_day    =31;
end_month  =12;
end_year   =2017;

%end_day    =1;
%end_month  =12;
%end_year   =2017;


%Runs (names) or experiments (numbers - starts with 1)
expt=[12,9,17,15];%2,5,7,10]
expt=[19,18];
expt=[30];

serie_or_maps=[0]; % 1 for serie, 2 for video, 3 for map, 0 for neither
my_dates=1;
inc_obs=1;

% Plot types
%plot_scatter=0;
plot_series =1;
plot_pdf    =0;
plot_psd    =0;
%plot_video  =0; 
plot_map    =0;
%plot_anim   =0;
save_fig    =1;
plt_show    =1;
interp_obs  =1 ;% only for SIE maps obs has 2x the model resolution

%Variables
vname ='sit_ross_sea'; %divergence'%'newice_mean_psd'; %'newice_perc'; % newice_perc 
varim =''; % unused 'sit' for model solo videos  % video



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% after 'BSOSE' run (ocean boundary cond), runs are all mEVP
runs={'50km_ocean_wind'      ,'50km_bsose_20180102'   ,'50km_hSnowAlb_20180102','50km_61IceAlb_20180102','50km_14kPmax_20180102',...       # 5
      '50km_20Clab_20180102' ,'50km_P14C20_20180102'  ,'50km_LandNeg2_20180102','50km_bsose_20130102'   ,'50km_dragWat01_20180102',...    # 10
      '50km_glorys_20180102' ,'BSOSE'                 ,'50km_mevp_20130102'    ,'50km_lemieux_20130102' ,'50km_h50_20130102',    ...       # 15
      '50km_hyle_20130102'   ,'50km_ckFFalse_20130102','50km_bWd020_20130102'  ,'mEVP+'                 ,'25km_bbm_20130102',   ...        # 20
      '25km_mevp_20130102'   ,'12km_bbm_20130102'     ,'12km_mEVP_20130102'    ,'50km_bWd016_20130102'  ,'50km_mCd01_20130102', ...        # 25
      '50km_bCd01_20130102'  ,'50km_mWd016_20130102'  ,'50km_10kPcom_20130102' ,'50km_mevp10kP_20130102','BBM',...    # '50km_b10kP2h_20130102', ...      # 30
      'mEVP'                 ,'50km_b14kP1h_20130102' ,'50km_m14kP1h_20130102' ,'50km_b14kP2h_20130102' ,'50km_m14kP2h_20130102',...       # 35
      '50km_mWd022_20130102' ,'50km_mWd024_20130102'}; %      # ,'50km_mevp10kP_20130102']#  ,'50km_bCd01_20130102']         # 33


expts=1:length(runs); %) #[0,1,2,3,4,5]

%Colors
colors={'r','b','k','r','m','b','y','g','r','b','k'};
obs_colors={'g','y','r'};

colorv=[0.8500 0.3250 0.0980; 0 0.4470 0.7410;
        1 0 0; 0 0 1];

% varrays according to vname
if strncmp(vname,'newice',6) 
  varray='newice'; 
elseif strncmp(vname,'divergence',10) 
  varray='siv'; 
elseif strncmp(vname,'sit',3) 
  varray='sit'; 
end

%trick to cover all months in runs longer than a year
end_month=end_month+1;
ym_start= 12*start_year + start_month - 1;
ym_end  = 12*end_year + end_month - 1;
end_month=end_month-1;


% SIE obs sources
obs_sources={'OSISAFease2'};%,'OSISAF-ease'] %['NSIDC','OSISAF','OSISAF-ease','OSISAFease2']: 
    
scrsz=[1 1 1366 768];
%scrsz=get(0,'screensize');

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
lon_mod = double(ncread(ncfile,'longitude')); %sit.to_masked_array() % Extract a given variable
lat_mod = double(ncread(ncfile,'latitude')); %sit.to_masked_array() % Extract a given variable
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


  if plot_series==1;

		if strcmp(vname,'sit_ross_sea') %|| strcmp(vname,'newice_mean_psd')

      % reading p-skrips data
      psk=open('/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/ross_sea/ross-sea-timeseries_daily_14_1.mat');
      pskfile='/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/ross_sea/mean_SIHEFF-2017-winter_runCase14.nc';
      lon_psk=double(ncread(pskfile,'XLONG'));
      lat_psk=double(ncread(pskfile,'XLAT'));
      time_psk=datenum(2017,2,1):datenum(2017,12,31);

 
      filepxx=[path_runs,run,'/output/ross_sea_sit_series_',datestr(time_mod(1),'yyyy-mm-dd'),'_',datestr(time_mod(end),'yyyy-mm-dd'),'.mat'];

      if exist(filepxx)==2
        display(['Loading: ',filepxx])
        load([filepxx])

      else
			  %% power spectrum analysis
			  display(['Computing ','daily average SIT from the Ross Sea'])
        k=0;
			  for i = times 
			    id=find(time_modi==i); k=k+1;
          mean_sit=squeeze(nanmean(vdatac(:,:,id),3));
			    display(['Interpolating ','daily average SIT from neXtSIM to the Ross Sea, day: ',datestr(i)])
          ross_sit=griddata(lon_nex,lat_nex,mean_sit,lon_psk,lat_psk);
          ross_sit_series(k)=nanmean(ross_sit(:));
        end

        display(['Saving: ',filepxx])
        save([filepxx],'ross_sit_series','times')

      end

      ross_sit_psk=nanmean(nanmean(psk.daily_all_thick_14,2),3);

			%close all
      figure('position',scrsz,'color',[1 1 1],'visible','on');  
      hold on
      plot(times,ross_sit_series,'k','linewidth',2)
      plot(time_psk,ross_sit_psk,'b','linewidth',2)
      legend('BBM','P-SKRIPS','location','best')
      datetick('x')

		end %  if strcmp(vname,'sit_ross_sea')

  end % plot_series



  if plot_map==1;

		if strcmp(vname,'newice_perc') || strcmp(vname,'newice_mean_psd')
      
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

      inan=find(pxx==0); pxx(inan)=nan; 
      if ke==1
       pxxst=pxx; 
      else
       pxx=pxx-pxxst; 
      end
			%% 
			% Original delta t = 6 hr, so that means delta f is cycles/6 hr. 
			% 1/f gives periods in 6hr/cylce. f/4 gives days/cycle
			% 2*pi/f*4 gives days
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


		elseif strcmp(vname,'sit_ross_sea') %|| strcmp(vname,'newice_mean_psd')

      % reading p-skrips data
      psk=open('/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/ross_sea/ross-sea-timeseries_daily_14_1.mat');
      pskfile='/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/ross_sea/mean_SIHEFF-2017-winter_runCase14.nc';
      lon_psk=ncread(pskfile,'XLONG');
      lat_psk=ncread(pskfile,'XLAT');
      time_psk=datenum(2017,2,1):datenum(2017,12,31);
 
			display(['Computing ','daily average SIT from the Ross Sea'])
      k=0;

      figure('position',scrsz,'color',[1 1 1],'visible','on');  
      load coastlines

			for i = times 
			  id=find(time_modi==i); k=k+1;
        mean_sit(:,:,k)=squeeze(nanmean(vdatac(:,:,id),3));

        exs=[1 2];
        for ex=exs

          if ex==3 % osisaf

          elseif ex==1 % p-skryps
            ip=find(time_psk==i); 
            model=squeeze(psk.daily_all_thick_14(ip,:,:))';
            lon_mod=lon_psk; lat_mod=lat_psk;
            mname='P-SKRIPS';
          elseif ex==2
            model=squeeze(mean_sit(:,:,k));
            lon_mod=lon_nex; lat_mod=lat_nex;
            mname=[run,' neXtSIM-Ant'];
          end
          model(model==0)=nan;

			    %close all
			    ax=subplot(1,2,ex);
          hold on; 
			    worldmap([-90 -60],[150 -130])

          projection = gcm;
          latlim = projection.maplatlimit;
          lonlim = [-130 150]; % projection.maplonlimit;
          if ~exist('gshhs_f.i', 'file');
              gshhs('gshhs_f.b', 'createindex');
          end
          % Load the GSHHG coastal polygon data version 2.3.7 - Full Resolution
          antarctica = gshhs('gshhs_f.b', latlim, lonlim);
          levels          = [antarctica.Level];
          land            = (levels == 1);
          lake            = (levels == 2);
          island          = (levels == 3); % island in a lake
          pond            = (levels == 4); % pond in an island in a lake
          ice_front       = (levels == 5); % ice shelves around Antarctica
          grounding_line  = (levels == 6); % land of Antarctica

          geoshow([antarctica(ice_front).Lat],      [antarctica(ice_front).Lon],      'DisplayType', 'Line', 'Color',[0 0 1]); % [230/255 230/255 230/255]); % gray
          %geoshow([antarctica(ice_front).Lat],      [antarctica(ice_front).Lon],      'DisplayType', 'Polygon', 'FaceColor',[0 0 1]); % [230/255 230/255 230/255]); % gray
          %geoshow([antarctica(grounding_line).Lat], [antarctica(grounding_line).Lon], 'DisplayType', 'Line',    'Color', [0 0 0]) %   [255/255 105/255 180/255]); % hot pink
          %geoshow([antarctica(land).Lat],           [antarctica(land).Lon],           'DisplayType', 'Polygon', 'FaceColor', [  0/255 100/255   0/255]); % forest green
          %geoshow([antarctica(lake).Lat],           [antarctica(lake).Lon],           'DisplayType', 'Polygon', 'FaceColor', [  0/255   0/255 128/255]); % navy blue
          %geoshow([antarctica(island).Lat],         [antarctica(island).Lon],         'DisplayType', 'Polygon', 'FaceColor', [210/255 105/255  30/255]); % chocolate
          %geoshow([antarctica(pond).Lat],           [antarctica(pond).Lon],           'DisplayType', 'Polygon', 'FaceColor', [ 84/255  84/255  84/255]); % light steel blue
          %setm(gca, 'FFaceColor', [.7 .7 .7]); % set background color (should be seas, oceans, bays, etc.)

			    pcolorm(double(lat_mod),double(lon_mod),model); 
			    set(gca,'clim',[0 3])
			    colormap(ax,cmocean('dense_r')); 
          colorbar
          h=title(['Sea ice thickness from ',mname,' on ',datestr(i,'YYYY-mm-dd')]);
          h.Position(2)=-6200000.36852015;
          plotm(coastlat,coastlon,'k')

        end

        if save_fig==1
          figname=[path_fig,run,'/map_',vname,'_',datestr(i,'yyyy-mm-dd'),'.png'];
          display(['Saving: ',figname]);
          export_fig(gcf,figname,'-png','-r150' );
          %print('-dpng','-r300',figname)
          %saveas(gcf,figname,'fig')
          clf('reset')
          set(gcf,'color',[1 1 1])
        end

      end % for i = times

		end %  if strcmp(vname,


  end % plot_map



  if plot_psd==1;

		if strcmp(vname,'newice_mean_psd')
      
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
              % PSD witouth windowing or anything here (2 degrees of freedom):
              [psd(i,j,:) fpsd] = PSD_ocen450(fld,4); % (time_mod(2)-time_mod(1)));
			  		end
			  	end
			  end
        display(['Saving: ',filepxx])
        save([filepxx],'pxx','f','psd','fpsd')
      end
			%% 
			% Original delta t = 6 hr, so that means delta f is cycles/6 hr. 
			% 1/f gives periods in 6hr/cylce. f/4 gives days/cycle
			% 2*pi/f*4 gives days
			T = 2*pi./(4*f); % (T is period in  days from the longest to the shortest period (longest = 256 days if input is 2 years. 1/2 day = 6h delta t ))
			Tpsd = 1./(fpsd); % (T is period in  days from the longest to the shortest period (longest = 256 days if input is 2 years. 1/2 day = 6h delta t ))
			ishort = find(T < 2,1); 
			ilong = find(T(2:end)<60,1); 
      T=fliplr(T);
      Tpsd=fliplr(Tpsd);

%return

      inan=find(pxx==0); pxx(inan)=nan; 
      inan=find(psd==0); psd(inan)=nan; 
      mpxx=nanmean(pxx,1); mpxx=nanmean(squeeze(mpxx),1); mpxx=flipud(mpxx);
      mpsd=nanmean(psd,1); mpsd=nanmean(squeeze(mpsd),1); mpsd=flipud(mpsd);
      scrsz=[1 1 1366 768];
      scrsz=get(0,'screensize');

			%close all
      colors=[0.8500 0.3250 0.0980; 0 0.4470 0.7410];
      if ke==1;
        ll=[];
        figure('position',scrsz,'color',[1 1 1],'visible','on'); 
        for i=1:length(expt);
          %loglog(T,mpxx,'color',colors(i,:),'linewidth',2); hold on
        end
      end
      ll=[ll,{run}];
      loglog(T,mpxx,'color',colors(ke,:),'linewidth',2); 
      %loglog(Tpsd,mpsd,'color',colors(ke,:),'linewidth',2); 
      hold on
      wpxx=(squeeze(pxx(100,200,:))); % somewhere in the Weddell
      wpsd=(squeeze(psd(100,200,:))); % somewhere in the Weddell
      loglog(T,wpxx,'--','color',colors(ke,:),'linewidth',2); hold on
      %loglog(Tpsd,wpsd,'--','color',colors(ke,:),'linewidth',2); hold on
      if ex==expt(end)
        set(gca,'fontsize',12,'fontweight','bold') 
        set(gca,'xdir','reverse') 
        %title(['Welch`s method PSD between ',datestr(time_mod(1),'yyyy-mm-dd'),' and ',datestr(time_mod(end),'yyyy-mm-dd')])
        title(['PSD between ',datestr(time_mod(1),'yyyy-mm-dd'),' and ',datestr(time_mod(end),'yyyy-mm-dd')])
        grid('on')
        %legend([ll])
        legend('BBM','BBM-Weddel','mEVP','mEVP-Weddel')
        xlabel('Period (day)')
        ylabel('New ice (m/day)^2')
        % saving fig
        if save_fig==1
          figname=[path_fig,run,'/psdw_',vname,'_',datestr(time_mod(1),'yyyy-mm-dd'),'_',datestr(time_mod(end),'yyyy-mm-dd'),'.png'];
          display(['Saving: ',figname]);
          %export_fig(gcf,figname,'-png','-r150' );
          print('-dpng','-r300',figname)
          %saveas(gcf,figname,'fig')
          %clf('reset')
          %set(gcf,'color',[1 1 1])
        end
      end

		end %  if strcmp(vname,'newice_mean_psd')

  end % if plot_psd==1;


  if plot_pdf==1;

		if strcmp(vname,'divergence')

          uc_mod = udatac.*3.6.*24;  vc_mod = vdatac.*3.6.*24;
          izero=find(uc_mod==0);uc_mod(izero)=nan; vc_mod(izero)=nan;

          dudx=(uc_mod(2:end,:,:)-uc_mod(1:end-1,:,:))./25; 
          dvdx=(vc_mod(2:end,:,:)-vc_mod(1:end-1,:,:))./25; 
          dudy=(uc_mod(:,2:end,:)-uc_mod(:,1:end-1,:))./25; 
          dvdy=(vc_mod(:,2:end,:)-vc_mod(:,1:end-1,:))./25; 

          div_mod=dudx(:,1:end-1,:)+dvdy(1:end-1,:,:); 
          shear_mod=sqrt( (dudx(:,1:end-1,:)+dvdy(1:end-1,:,:)).^2 + (dudy(2:end,:,:)+dvdx(:,2:end,:)).^2 );
          con_mod=div_mod; inan=find(con_mod>0); con_mod(inan)=nan;
          inan=find(div_mod<0); div_mod(inan)=nan;

          %hist_int=2E-2;

          %d=div_mod(:);
          %y=histogram(d,'Normalization','pdf');
          %close; 
          %loglog(y.Data,'color',colors(ke,:),'linewidth',2); 

          if ke==1;
            ll=[];
            figure('position',scrsz,'color',[1 1 1],'visible','on'); 
            for i=1:length(expt);
              %loglog(T,mpxx,'color',colors(i,:),'linewidth',2); hold on
            end
          end
          ll=[ll,{run}];

          nbins = 100; 
          binedges = [0 logspace(-3,0,nbins)];
          bincenters = binedges(1:end-1) + 0.5*diff(binedges); 
          dx = diff(binedges); 
          [a_div,~] = histcounts(div_mod(:),binedges);
          [a_shear,~] = histcounts(shear_mod(:),binedges);
          ndiv = nansum(a_div); 
          nshear = nansum(a_shear);
          ndiv2=a_div.*bincenters./dx;%,'reverse');



          subplot(131)
          loglog(bincenters,ndiv2,'color',colorv(ke,:),'linewidth',2); 
          hold on
          %loglog(bincenters,a_shear); 
          title('$N\cdot f(x)\cdot dx$','Interpreter','latex')
          grid on; box on; 
          %xlim([1e-3 1])
          %ylim([10 1e6])

          subplot(132)
          loglog(bincenters,a_div./dx,'color',colorv(ke,:),'linewidth',2); 
          hold on
          %loglog(bincenters,a_shear./dx); 
          grid on; box on; 
          %xlim([1e-3 1])
          %ylim([1e2 1e9])
          title('$N\cdot f(x)$','Interpreter','latex')
          
          subplot(133)
          loglog(bincenters,a_div./dx./ndiv,'color',colorv(ke,:),'linewidth',2); 
          hold on
          %loglog(bincenters,a_shear./dx./nshear); 
          grid on; box on; 
          %xlim([1e-3 1])
          %ylim([1e-3 4e2])
          title('$f(x)$','Interpreter','latex')

          if ex==expt(end)
            %set(gca,'fontsize',12,'fontweight','bold') 
            %set(gca,'xdir','reverse') 
            %title(['Welch`s method PSD between ',datestr(time_mod(1),'yyyy-mm-dd'),' and ',datestr(time_mod(end),'yyyy-mm-dd')])
            %title(['PSD between ',datestr(time_mod(1),'yyyy-mm-dd'),' and ',datestr(time_mod(end),'yyyy-mm-dd')])
            %grid('on')
            legend([ll])
            %legend('mEVP','BBM')
            %xlabel('Period (day)')
            %ylabel('New ice (m/day)^2')

            % saving fig
            if save_fig==5
              figname=[path_fig,run,'/psdw_',vname,'_',datestr(time_mod(1),'yyyy-mm-dd'),'_',datestr(time_mod(end),'yyyy-mm-dd'),'.png'];
              display(['Saving: ',figname]);
              %export_fig(gcf,figname,'-png','-r150' );
              print('-dpng','-r300',figname)
              %saveas(gcf,figname,'fig')
              %clf('reset')
              %set(gcf,'color',[1 1 1])
            end

          end

		end %  if strcmp(vname,'bla')

bla=1;
if bla==2

    % FIGURE
    scrsz=[1 1 1366 768];
    scrsz=get(0,'screensize');

		%close all
    colors=[0.8500 0.3250 0.0980; 0 0.4470 0.7410];
    if ke==1;
      ll=[];
      figure('position',scrsz,'color',[1 1 1],'visible','on'); 
      for i=1:length(expt);
        %loglog(T,mpxx,'color',colors(i,:),'linewidth',2); hold on
      end
    end
    ll=[ll,{run}];
    loglog(T,mpxx,'color',colors(ke,:),'linewidth',2); 
    %loglog(Tpsd,mpsd,'color',colors(ke,:),'linewidth',2); 
    hold on
    wpxx=(squeeze(pxx(100,200,:))); % somewhere in the Weddell
    wpsd=(squeeze(psd(100,200,:))); % somewhere in the Weddell
    loglog(T,wpxx,'--','color',colors(ke,:),'linewidth',2); hold on
    %loglog(Tpsd,wpsd,'--','color',colors(ke,:),'linewidth',2); hold on
    if ex==expt(end)
      set(gca,'fontsize',12,'fontweight','bold') 
      set(gca,'xdir','reverse') 
      %title(['Welch`s method PSD between ',datestr(time_mod(1),'yyyy-mm-dd'),' and ',datestr(time_mod(end),'yyyy-mm-dd')])
      title(['PSD between ',datestr(time_mod(1),'yyyy-mm-dd'),' and ',datestr(time_mod(end),'yyyy-mm-dd')])
      grid('on')
      %legend([ll])
      legend('BBM','BBM-Weddel','mEVP','mEVP-Weddel')
      xlabel('Period (day)')
      ylabel('New ice (m/day)^2')

      % saving fig
      if save_fig==1
        figname=[path_fig,run,'/psdw_',vname,'_',datestr(time_mod(1),'yyyy-mm-dd'),'_',datestr(time_mod(end),'yyyy-mm-dd'),'.png'];
        display(['Saving: ',figname]);
        %export_fig(gcf,figname,'-png','-r150' );
        print('-dpng','-r300',figname)
        %saveas(gcf,figname,'fig')
        %clf('reset')
        %set(gcf,'color',[1 1 1])
      end

    end

end % if bla==2

  end % if plot_pdf==1;

end % loop in expts

display('End of script')
%%plt.close('all')
