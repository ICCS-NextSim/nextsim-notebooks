from roms_tools import Grid, ROMSOutput, InitialConditions

from datetime import datetime

grid_path='/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/'

grid_file='roms_grid_33km_lo90_la89.nc' #_with_iceshelves.nc'

grid_file="roms_etopo_grid_29km_lo90_la90_nx392_size_x11600_with_iceshelves.nc"

grid = Grid.from_file(grid_path+grid_file)

ini_time = datetime(2013, 1, 2, 0, 0, 0) # noon on January 2, 2012

date_str = ini_time.strftime("%Y-%m-%d")

ini_path = "/oscar/data/deeps/private/chorvat/data/GLORYS/3D-southern-ocean/"

ini_file=f"cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-30.00S_0.49-5727.92m_{date_str}.nc"

ini_file=f"cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-30.00S_0.49-5727.92m_2013-01-01-2013-01-31.nc"


initial_conditions = InitialConditions(
    grid=grid,
    ini_time=ini_time,
    source={"name": "GLORYS", "path": ini_path+ini_file},
    model_reference_date=datetime(2013, 1, 1), # this is the default
    adjust_depth_for_sea_surface_height = True,
    use_dask=True,
    allow_flex_time= True
)


initial_conditions.plot("temp", s=-1)  # plot uppermost layer

#initial_conditions.plot("temp", s=0, depth_contours=True)  # plot bottom layer
#
#initial_conditions.plot("temp", xi=0, layer_contours=True)

#initial_conditions.plot("temp", eta=0, xi=0)

#initial_conditions.plot("u", s=-1)  # plot uppermost layer

#initial_conditions.plot("u", eta=0)




