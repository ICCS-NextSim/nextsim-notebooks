from roms_tools import Grid, BoundaryForcing

from datetime import datetime

grid_path='/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/'
grid_file='roms_grid_33km_lo90_la89.nc' #_with_iceshelves.nc'
grid_file="roms_etopo_grid_29km_lo90_la90_nx392_size_x11600_with_iceshelves.nc"

grid = Grid.from_file(grid_path+grid_file)

start_time = datetime(2013,1,2,12,00,00)
end_time = datetime(2013,1,3,12,00,00)

glorys_path = "/oscar/data/deeps/private/chorvat/data/GLORYS/3D-southern-ocean/cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-30.00S_0.49-5727.92m_2013-01-01-2013-01-31.nc"

#ini_file=f"cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-30.00S_0.49-5727.92m_{date_str}.nc"


boundary_forcing = BoundaryForcing(
    grid=grid,
    start_time=start_time,
    end_time=end_time,
    boundaries={
        "south": True,
        "east": True,
        "north": True,  # northern boundary excluded
        "west": True,
    },
    source={"name": "GLORYS", "path": glorys_path},
    type="physics",  # "physics" or "bgc"; default is "physics"
    model_reference_date=datetime(2013, 1, 1), # this is the default
    use_dask=True,
    apply_2d_horizontal_fill = True
)



