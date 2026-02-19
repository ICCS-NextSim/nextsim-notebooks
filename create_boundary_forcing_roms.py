from roms_tools import Grid, BoundaryForcing
from datetime import datetime
import matplotlib.pyplot as plt
plt.ion()
plt.close('all')

start_time = datetime(2013,1,2,12,00,00)
end_time = datetime(2013,1,30,12,00,00)

date_str = start_time.strftime("%Y-%m-%d")
date_end = end_time.strftime("%Y-%m-%d")

grid_path='/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/'
grid_file="roms_etopo_grid_26km.nc"

bry_path='/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/boundary_26km/'
bry_file=f'bry_glorys_roms_etopo_grid_26km_{date_str}_{date_end}.nc'
bry_file=bry_path+bry_file
print(bry_file)

grid = Grid.from_file(grid_path+grid_file)

glorys_path = "/oscar/data/deeps/private/chorvat/data/GLORYS/3D-southern-ocean/cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-20.00S_0.49-5727.92m_2013-01-01-2013-01-31.nc"

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
    #apply_2d_horizontal_fill = True
)


boundary_forcing.save(bry_file, group=True)

