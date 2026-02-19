% converts a ROMS grid to a neXtSIM grid
% it uses mkCplMesh.m and cpl_coords.m which were provided by Einar Olason
% these two script were written for converting NEMO or TOPAZ grids into
% neXtSIM grids
% the approach here is to tweak a ROMS grid so it can mimic a NEMO grid,
% both frameworks are similar enough


clear all
close all

%% set paths
wrk_dir = '/oscar/data/deeps/private/chorvat/santanarc/mapx_install/scripts/'; %[realpath('~/OneDrive/00_shared/015_neXtSIM_ROMS/') '/'];

inp_pth = '/oscar/data/deeps/private/chorvat/santanarc/n/southern/mesh/';
out_pth = [inp_pth];

%% include necessary libraries
%addpath([wrk_dir 'mlb_scripts'])
addpath(genpath('/oscar/data/deeps/private/chorvat/santanarc/n/nextsim-tools/matlab/'));
addpath('/oscar/data/deeps/private/chorvat/santanarc/mapx_install/mlb_lib_jes/')
% ROMS grid

src_fle_nme = 'roms_his_26km_etopo.nc'; % 'roms_etopo_grid_29km.nc';

%% read ROMS grid

src_grid = [inp_pth src_fle_nme];

mkCplMesh_rcs_oscar('ROMS',0,src_grid)



