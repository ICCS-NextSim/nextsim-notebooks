#Script to create a southern ocean roms grid

#Use GSHHS_shp/h/GSHHS_h_L5.shp, Antarctic shapefile, to include iceshelf in the grid. Run with and without it. Then merge masks later with modify_roms_grid.py

from roms_tools import Grid
from sys import exit
import os

Grid.N=75
Grid.hmin=20
Grid.hc=300 #  width of surface or bottom boundary layer in which higher vertical resolution (levels) is required during stretching.
Grid.theta_s=5 
Grid.theta_b=2

# SRTM15
nx=260
size_x=8600
lo=90 # 0, -90, -180, 180, 
la=89

# etopo5
nx=392 # 370
size_x=11600
lo=90 # 0, -90, -180, 180, 
la=90

fileout=f'/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/roms_etopo_grid_{int(size_x/nx)}km_lo{lo}_la{la}_nx{nx}_size_x{size_x}_with_iceshelves.nc'

print(fileout)

grid = Grid(
    nx=nx,  # number of grid points in x-direction
    ny=nx+2,  # number of grid points in y-direction
    size_x=size_x,  # domain size in x-direction (in km)
    size_y=size_x,  # domain size in y-direction (in km)
    center_lon=lo,  # longitude of the center of the domain # -90,
    center_lat=-la,  # latitude of the center of the domain
    rot=0,  # rotation of the grid (in degrees)
    #topography_source={
    #    "name": "SRTM15",
    #    "path": "/oscar/data/deeps/private/chorvat/data/SRTM15/SRTM15_V2.6.nc",
    #},
    mask_shapefile="/oscar/data/deeps/private/chorvat/data/GSHHG/GSHHS_shp/h/GSHHS_h_L5.shp", # Antarctic shapefile, used to include iceshelf. Run with and without and merge masks later with modify_roms_grid.py
    verbose=True,
    N=75,  # number of vertical layers
    hmin = 20.,
    theta_s = 5,
    theta_b = 2,
    hc = 100,

)

#grid.plot()

#exit()

if os.path.exists(fileout):
  os.remove(fileout)
  print(f"Deleted: {fileout}")

grid.save(fileout)


