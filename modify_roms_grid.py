import xarray as xr
import os
import shutil
from sys import exit


filepath='/oscar/data/deeps/private/chorvat/santanarc/n/southern/roms_sim/input/'

bkpfile = filepath+"roms_grid_33km_lo90_la89_with_iceshelves.nc.bkp"
origfile = filepath+"roms_grid_33km_lo90_la89_with_iceshelves.nc"
extrafile = filepath+"roms_grid_33km_lo90_la89.nc"

fname="roms_etopo_grid_26km_lo90_la90_nx378_size_x10000"

bkpfile = filepath+fname+"_with_iceshelves.nc.bkp"
origfile = filepath+fname+"_with_iceshelves.nc"
extrafile = filepath+fname+".nc"

# copy bkp file to orig file
# Restore original from backup BEFORE modification
shutil.copy2(bkpfile, origfile)
print("Backup restored to original file.")

ds = xr.open_dataset(origfile, mode="r+")
dsex = xr.open_dataset(extrafile, mode="r")

varnames = ["mask_rho","mask_v", "mask_u"]

for varname in varnames:
  # Add masks
  combined_mask = ds[varname] + dsex[varname]
  
  # Convert to binary mask: >=1 -> 1, <1 -> 0
  ds[varname] = xr.where(combined_mask > 1, 1, 0)



# ---------------------------------------------------------------------
# Add xl and el (scalar variables)
# ---------------------------------------------------------------------
dx = 1.0 / ds["pm"]
dy = 1.0 / ds["pn"]

ds["xl"] = xr.DataArray(
    dx.sum(dim="xi_rho").mean(dim="eta_rho"),
    attrs={"long_name": "Length of domain in XI-direction", "units": "meter"}
)

ds["el"] = xr.DataArray(
    dy.sum(dim="eta_rho").mean(dim="xi_rho"),
    attrs={"long_name": "Length of domain in ETA-direction", "units": "meter"}
)

# ---------------------------------------------------------------------
# Compute dndx and dmde (ROMS formulation)
#   dndx = - (1/pn) * ∂pn/∂x
#   dmde = - (1/pm) * ∂pm/∂y
# ---------------------------------------------------------------------
pn = ds["pn"]
pm = ds["pm"]

# Grid metrics
dndx = - (1.0 / pn) * pn.differentiate("xi_rho")
dmde = - (1.0 / pm) * pm.differentiate("eta_rho")

ds["dndx"] = dndx.assign_attrs(
    long_name="xi-derivative of inverse metric pn",
    units="meter-1"
)

ds["dmde"] = dmde.assign_attrs(
    long_name="eta-derivative of inverse metric pm",
    units="meter-1"
)



#mask_psi

rmask = ds["mask_rho"]

# Compute mask_psi exactly like uvp_masks.m
mask_psi = (
    rmask.isel(eta_rho=slice(0, -1), xi_rho=slice(0, -1)) *
    rmask.isel(eta_rho=slice(1,  None), xi_rho=slice(0, -1)) *
    rmask.isel(eta_rho=slice(0, -1), xi_rho=slice(1,  None)) *
    rmask.isel(eta_rho=slice(1,  None), xi_rho=slice(1,  None))
)

# Rename dimensions to psi-grid
mask_psi = mask_psi.rename(
    {"eta_rho": "eta_psi", "xi_rho": "xi_psi"}
)

# Assign attributes
ds["mask_psi"] = mask_psi.assign_attrs(
    long_name="mask on PSI-points",
    flag_values=[0, 1],
    flag_meanings="land water"
)

# Optional sanity check
print("mask_psi min/max:",
      ds["mask_psi"].min().values,
      ds["mask_psi"].max().values)



##############

os.remove(origfile)
print("Writing file: "+origfile)
ds.to_netcdf(origfile, mode="a")

ds.close()
dsex.close()

