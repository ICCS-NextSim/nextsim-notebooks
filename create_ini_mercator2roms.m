%
%  D_MERCATOR2ROMS:  Driver script to create a ROMS initial conditions
%
%  This a user modifiable script that can be used to prepare ROMS
%  initial conditions NetCDF file from Mercator dataset. It sets-up
%  all the necessary parameters and variables. USERS can use this
%  as a prototype for their application.
%

% svn $Id$
%=========================================================================%
%  Copyright (c) 2002-2025 The ROMS Group                                 %
%    Licensed under a MIT/X style license                                 %
%    See License_ROMS.md                            Hernan G. Arango      %
%=========================================================================%

clear
close all
tic

% Set file names.

%OPR_Dir = '/home/arango/ocean/toms/repository/Projects/philex/Mercator/OPR';
RTR_Dir = '/oscar/data/deeps/private/chorvat/data/GLORYS/3D-southern-ocean/'; % path for ini cond data source

Tfile = 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-20.00S_0.49-5727.92m_2013-01-02.nc';
Ufile = Tfile; % 'ext-PSY3V2R2_1dAV_20070101_20070102_gridU_R20070110.nc.gz';
Vfile = Tfile; % 'ext-PSY3V2R2_1dAV_20070101_20070102_gridV_R20070110.nc.gz';

GRDpath='/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/';
GRDname = 'roms_etopo_grid_26km.nc';
GRDname = [GRDpath,GRDname];

INIpath='/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/ini_26km/';
INIname = [INIpath,'ini_2013-01-02_roms_etopo_grid_26km.nc'];

CREATE = true;                   % logical switch to create NetCDF
report = false;                  % report vertical grid information

% Get number of grid points.

[Lr,Mr]=size(nc_read([GRDname],'h'));

Lu = Lr-1;   Lv = Lr;
Mu = Mr;     Mv = Mr-1;


%--------------------------------------------------------------------------
%  Create initial conditions NetCDF file.
%--------------------------------------------------------------------------

% Set full path of Mercator files assigned as initial conditions.

fileT=fullfile(RTR_Dir,Tfile); lenT=length(fileT);
fileU=fullfile(RTR_Dir,Ufile); lenU=length(fileU);
fileV=fullfile(RTR_Dir,Vfile); lenV=length(fileV);


%--------------------------------------------------------------------------
%  Set application parameters in structure array, S.
%--------------------------------------------------------------------------

S.ncname      = INIname;     % output NetCDF file

S.spherical   = 1;           % spherical grid

S.Lm          = Lr-2;        % number of interior RHO-points, X-direction
S.Mm          = Mr-2;        % number of interior RHO-points, Y-direction
S.N           = 75;          % number of vertical levels at RHO-points
S.NT          = 2;           % total number of tracers

S.Vtransform  = 2;           % vertical transfomation equation
S.Vstretching = 4;           % vertical stretching function

S.theta_s     = 5.0;         % S-coordinate surface control parameter
S.theta_b     = 2.0;         % S-coordinate bottom control parameter
S.Tcline      = 100.0;       % S-coordinate surface/bottom stretching width
S.hc          = S.Tcline;    % S-coordinate stretching width

%--------------------------------------------------------------------------
%  Set grid variables.
%--------------------------------------------------------------------------

S.h           = nc_read(GRDname, 'h');            % bathymetry

S.lon_rho     = nc_read(GRDname, 'lon_rho');      % RHO-longitude
S.lat_rho     = nc_read(GRDname, 'lat_rho');      % RHO-latitude

S.lon_u       = nc_read(GRDname, 'lon_u');        % U-longitude
S.lat_u       = nc_read(GRDname, 'lat_u');        % U-latitude

S.lon_v       = nc_read(GRDname, 'lon_v');        % V-longitude
S.lat_v       = nc_read(GRDname, 'lat_v');        % V-latitude

S.mask_rho    = nc_read(GRDname, 'mask_rho');     % RHO-mask
S.mask_u      = nc_read(GRDname, 'mask_u');       % U-mask
S.mask_v      = nc_read(GRDname, 'mask_v');       % V-mask

S.angle       = nc_read(GRDname, 'angle');        % curvilinear angle

%  Set vertical grid variables.

kgrid=0;                                          % RHO-points

[S.s_rho, S.Cs_r]=stretching(S.Vstretching, ...
                             S.theta_s, S.theta_b, S.hc, S.N,         ...
                             kgrid, report);

kgrid=1;                                          % W-points			 

[S.s_w,   S.Cs_w]=stretching(S.Vstretching, ...
                             S.theta_s, S.theta_b, S.hc, S.N,         ...
                             kgrid, report);

%--------------------------------------------------------------------------
%  Interpolate initial conditions from Mercator data to application grid.
%--------------------------------------------------------------------------

disp(' ')
disp(['Interpolating from Mercator to ROMS grid ...']);
disp(' ')

%  Uncompress input Mercator files.

%s=unix(['gunzip ',fileT]);
%s=unix(['gunzip ',fileU]);
%s=unix(['gunzip ',fileV]);

%  Read Mercator data has a time coordinate counter (seconds) that
%  starts on 11-Oct-2006.

time=nc_read(fileT(1:lenT),'time');
%mydate=datestr(datenum('11-Oct-2006')+time/86400-0.5,0);
mydate=datestr(datenum('01-Jan-1950')+time/(24),0); % Hours Since 1950-01-01

%  Set initial conditions time (seconds). The time coordinate for this
%  ROMS application is "seconds since 2007-01-01 00:00:00". The 0.5
%  coefficient here is to account Mecator daily average.

%MyTime=time/86400-(datenum('1-Jan-2007')-datenum('11-Oct-2006'))-0.5;
TIME_REF=datenum('1-Jan-2013'); % this date should be equal to TIME_REF in the ocean.in file
MyTime=time/24-(TIME_REF-datenum('01-Jan-1950')); % 

disp([ '    Processing: ', mydate]);
disp(' ')
  
%  Get Mercator grid.

Tlon=nc_read(fileT(1:lenT),'longitude');
Tlat=nc_read(fileT(1:lenT),'latitude');
Tdepth=nc_read(fileT(1:lenT),'depth');

%  In the western Pacific, the level 50 (z=5727.9 m) of the Mercator data
%  is all zeros. Our grid needs depth of Zr=-5920 m.  Therefore, the depths
%  are modified in level 49 (z=5274.7 m) to bound the vertical interpolation.

Tdepth(49)=6500; Udepth(49)=6500; Vdepth(49)=6500;
Tdepth(50)=6700; Udepth(50)=6700; Vdepth(50)=6700;

% circshifting lon

Tlong=Tlon;
Tlongwrap = wrapTo360(Tlong);
crossingidx = find(diff(Tlongwrap) < 0, 1);
Tlon = circshift(Tlongwrap(:),-crossingidx,1);
%Bshifted = circshift(B, -crossingidx, 1); circshifting variables

% Making Lat go to the south pole 

%Tlatn=vertcat([-90:.25:-80.25]',Tlat);
%zerom=zeros(length(Tlon),length(Tlatn)-length(Tlat)); % creating zero to pad in glorys' variables
%Tlat=Tlatn;


Ulon=Tlon; % nc_read(fileU(1:lenU),'nav_lon');
Ulat=Tlat; %nc_read(fileU(1:lenU),'nav_lat');
Udepth=Tdepth; % nc_read(fileU(1:lenU),'depthu');

Vlon=Tlon; % nc_read(fileU(1:lenU),'nav_lon');
Vlat=Tlat; %nc_read(fileU(1:lenU),'nav_lat');
Vdepth=Tdepth; % nc_read(fileU(1:lenU),'depthu');

%  Read in initial conditions fields.

Zeta=nc_read(fileT(1:lenT),'zos');
Temp=nc_read(fileT(1:lenT),'thetao');
Salt=nc_read(fileT(1:lenT),'so');
Uvel=nc_read(fileU(1:lenU),'uo');
Vvel=nc_read(fileV(1:lenV),'vo');

%Uvel(:)=0; Vvel(:)=0;

% Padding variables with zeros at the south pole

%Zetam=cat(2,zerom,Zeta);
%Zetam=Zeta;
Zeta = circshift(Zeta, -crossingidx, 1);
Temp = circshift(Temp, -crossingidx, 1);
Salt = circshift(Salt, -crossingidx, 1);
Uvel = circshift(Vvel, -crossingidx, 1);
Vvel = circshift(Vvel, -crossingidx, 1);

%  Determine Mercator Land/Sea mask.  Since Mercator is a Z-level
%  model, the mask is 3D.

%Rmask2d=ones(size(Zeta));
%ind=find(Zetam == 0);
%Rmask2d(ind)=0;
Rmask3d=ones(size(Temp));
ind=find(Temp == 0);
Rmask3d(ind)=0;

clear ind

%  Compress input Mercator files.

%s=unix(['gzip ',fileT(1:lenT)]);
%s=unix(['gzip ',fileU(1:lenU)]);
%s=unix(['gzip ',fileV(1:lenV)]);

%  Set initial conditions time (seconds). The time coordinate for this
%  ROMS application is "seconds since 2007-01-01 00:00:00". The 0.5
%  coefficient here is to account Mecator daily average.

% MyTime is defined above
%MyTime=time/86400-(datenum('1-Jan-2007')-datenum('11-Oct-2006'))-0.5;

S.ocean_time = MyTime*86400;
% S.ocean_time = 86400;               % set to Jan 1, because of forcing

%  Interpolate free-surface initial conditions.

[Tlonm,Tlatm]=meshgrid(Tlon,Tlat'); Tlonm=Tlonm'; Tlatm=Tlatm';
Ulonm=Tlonm; Ulatm=Tlatm;
Vlonm=Tlonm; Vlatm=Tlatm;

toc
zeta=mercator2roms('zeta',S,Zeta,Tlonm,Tlatm,Rmask3d(:,:,1));
%zeta=zeros(size(S.lon_rho));
toc

%figure; pcolor(Tlonm,Tlatm,Zeta); shading flat; colorbar
%figure; pcolor(zeta); shading flat; colorbar
%return

%  Compute ROMS model depths.  Ignore free-sruface contribution
%  so interpolation is bounded below mean sea level.

ssh=zeros(size(zeta));

igrid=1;
[S.z_r]=set_depth(S.Vtransform, S.Vstretching,                        ...
                  S.theta_s, S.theta_b, S.hc, S.N,                    ...
		  igrid, S.h, ssh, report);
	      
igrid=3;
[S.z_u]=set_depth(S.Vtransform, S.Vstretching,                        ...
                  S.theta_s, S.theta_b, S.hc, S.N,                    ...
		  igrid, S.h, ssh, report);

igrid=4;
[S.z_v]=set_depth(S.Vtransform, S.Vstretching,                        ...
                  S.theta_s, S.theta_b, S.hc, S.N,                    ...
		  igrid, S.h, ssh, report);

%  Compute ROMS vertical level thicknesses (m).
	      
N=S.N;
igrid=5;
[S.z_w]=set_depth(S.Vtransform, S.Vstretching,                        ...
                  S.theta_s, S.theta_b, S.hc, S.N,                    ...
		  igrid, S.h, zeta, report);

S.Hz=S.z_w(:,:,2:N+1)-S.z_w(:,:,1:N);
	      
%  Interpolate temperature and salinity.

temp=mercator2roms('temp',S,Temp,Tlonm,Tlatm,Rmask3d,Tdepth);
toc
salt=mercator2roms('salt',S,Salt,Tlonm,Tlatm,Rmask3d,Tdepth);
toc
Urho=mercator2roms('u'   ,S,Uvel,Ulonm,Ulatm,Rmask3d,Udepth);
toc
Vrho=mercator2roms('v'   ,S,Vvel,Vlonm,Vlatm,Rmask3d,Vdepth);
toc

%salt=zeros(size(temp));
%Urho=zeros(size(temp));
%Vrho=zeros(size(temp));

% iReplacing remaining NaN points with the next data

sumnan=sum(isnan(temp(:)))+sum(isnan(salt(:)))+sum(isnan(Urho(:)))+sum(isnan(Vrho(:)))+sum(isnan(zeta(:)));

while sumnan>0

  inan=isnan(zeta); indn=find(inan==1); 
  if ~isempty(indn) 
  	for i=indn; zeta(i)=zeta(i+1); end; 
  end

  inan=isnan(temp); indn=find(inan==1); 
  if ~isempty(indn) 
  	for i=indn; temp(i)=temp(i+1); end; 
  end

  inan=isnan(salt); indn=find(inan==1); 
  if ~isempty(indn) 
    for i=indn; salt(i)=salt(i+1); end
  end

  inan=isnan(Urho); indn=find(inan==1); 
  if ~isempty(indn) 
    for i=indn; Urho(i)=Urho(i+1); end; 
  end

  inan=isnan(Vrho); indn=find(inan==1); 
  if ~isempty(indn) 
    for i=indn; Vrho(i)=Vrho(i+1); end
  end
  
  sumnan=sum(sum(isnan(temp(:))))+sum(sum(isnan(salt(:))))+sum(sum(isnan(Urho(:))))+sum(sum(isnan(Vrho(:))))+sum(isnan(zeta(:)));

end

%return


%  Process velocity: rotate and/or average to staggered C-grid locations.

[u,v]=roms_vectors(Urho,Vrho,S.angle,S.mask_u,S.mask_v);

%  Compute barotropic velocities by vertically integrating (u,v).

[ubar,vbar]=uv_barotropic(u,v,S.Hz);

%--------------------------------------------------------------------------
%  Create initial condition Netcdf file.
%--------------------------------------------------------------------------

if (CREATE),
  [status]=c_initial(S);

%  Set attributes for "ocean_time".

  avalue='seconds since 2013-01-01 00:00:00';
  [status]=nc_attadd(INIname,'units',avalue,'ocean_time');
  
  avalue='gregorian';
  [status]=nc_attadd(INIname,'calendar',avalue,'ocean_time');

%  Set global attributes.

  avalue='Southern Ocean 26 km';
  [status]=nc_attadd(INIname,'title',avalue);

  avalue='Mercator GLORYS daily average, 0.08 degree resolution';
  [status]=nc_attadd(INIname,'data_source',avalue);

  [status]=nc_attadd(INIname,'grd_file',GRDname);
end,

%--------------------------------------------------------------------------
%  Write out initial conditions.
%--------------------------------------------------------------------------

if (CREATE),
  disp(' ')
  disp([ 'Writing initial conditions ...']);
  disp(' ')

  [status]=nc_write(INIname, 'spherical',   S.spherical);
  [status]=nc_write(INIname, 'Vtransform',  S.Vtransform);
  [status]=nc_write(INIname, 'Vstretching', S.Vstretching);
  [status]=nc_write(INIname, 'theta_s',     S.theta_s);
  [status]=nc_write(INIname, 'theta_b',     S.theta_b);
  [status]=nc_write(INIname, 'Tcline',      S.Tcline);
  [status]=nc_write(INIname, 'hc',          S.hc);
  [status]=nc_write(INIname, 's_rho',       S.s_rho);
  [status]=nc_write(INIname, 's_w',         S.s_w);
  [status]=nc_write(INIname, 'Cs_r',        S.Cs_r);
  [status]=nc_write(INIname, 'Cs_w',        S.Cs_w);

  [status]=nc_write(INIname, 'h',           S.h);
  [status]=nc_write(INIname, 'lon_rho',     S.lon_rho);
  [status]=nc_write(INIname, 'lat_rho',     S.lat_rho);
  [status]=nc_write(INIname, 'lon_u',       S.lon_u);
  [status]=nc_write(INIname, 'lat_u',       S.lat_u);
  [status]=nc_write(INIname, 'lon_v',       S.lon_v);
  [status]=nc_write(INIname, 'lat_v',       S.lat_v);
  
  IniRec = 1;

  [status]=nc_write(INIname, 'ocean_time', S.ocean_time, IniRec);

  [status]=nc_write(INIname, 'zeta', zeta, IniRec);
  [status]=nc_write(INIname, 'ubar', ubar, IniRec);
  [status]=nc_write(INIname, 'vbar', vbar, IniRec);
  [status]=nc_write(INIname, 'u',    u,    IniRec);
  [status]=nc_write(INIname, 'v',    v,    IniRec);
  [status]=nc_write(INIname, 'temp', temp, IniRec);
  [status]=nc_write(INIname, 'salt', salt, IniRec);
end,

%--------------------------------------------------------------------------
%  Set masking indices to facilitate plotting.  They can be used to
%  replace ROMS Land/Sea mask with NaNs.
%--------------------------------------------------------------------------

inr2d=find(S.mask_rho == 0);
inr3d=find(repmat(S.mask_rho,[1,1,N]) == 0);
toc
