#!/usr/bin/env python
import sys
import os
import numpy as np
import argparse
from netCDF4 import Dataset


def get_parser():
    """Create parser for command line inputs"""
    parser = argparse.ArgumentParser(
        description="Add a cyclic longitude column to GLORYS12 NetCDF and replace missing values"
    )
    parser.add_argument("input_file", type=str, help="File to be processed")
    parser.add_argument("outdir", type=str, help="Where to save the output file")
    return parser


class MakeGlorys12Cyclic:
    """Add an extra lon column to GLORYS12 NetCDF file"""

    def __init__(self, input_file, outdir):
        self.input_file = input_file
        self.outdir = outdir

    @staticmethod
    def copy_extend_netcdf_file(src_ds, dst_ds):
        """
        Copy dataset, extend longitude by 1, and modify or fill missing values per variable.
        """

        # Dictionary: desired replacement for missing values (per variable)
        fill_values = {
            "thetao": 0.0,
            "so": 33.9758, # 34.0,
            "uo": 0.0,
            "vo": 0.0,
            "zos": -0.860163, # 0.0,
            "mlotst": 22.1078, # 0.0,
            "siconc": 0.0,
            "sithick": 0.0,
        }

        # Copy global attributes
        dst_ds.setncatts(vars(src_ds))

        # Copy dimensions (extend longitude by one)
        for d in src_ds.dimensions.values():
            sz = d.size
            if d.isunlimited():
                sz = None
            if d.name == "longitude":
                sz += 1
            dst_ds.createDimension(d.name, sz)

        # Copy and modify variables
        for src_var in src_ds.variables.values():
            vname = src_var.name
            print(f"Processing {vname}")

            dst_var = dst_ds.createVariable(
                src_var.name, src_var.dtype, src_var.dimensions #, zlib=True
            )
            dst_var.setncatts(vars(src_var))

            v = src_var[:]

            # --- Modify thetao south of -70° latitude ---
            if vname == "thetao" and "latitude" in src_ds.variables:
                lat = src_ds.variables["latitude"][:]
                lat_mask = lat < -74
                v[..., lat_mask, :] = -1.0
                print("Set thetao south of -70° latitude to -2°C")
            # ---------------------------------------------

            # --- Replace missing or masked values per variable ---
            if np.ma.isMaskedArray(v):
                # Fill masked array using the variable's fill value if available
                fill_val = fill_values.get(vname, np.nan)
                v = v.filled(fill_val)
                print(f"Filled masked {vname} with {fill_val}")
            else:
                # Replace NaNs if not a masked array
                fill_val = fill_values.get(vname, np.nan)
                nan_count = np.isnan(v).sum()
                if nan_count > 0:
                    print(f"Found {nan_count} NaNs in {vname}, replacing with {fill_val}")
                    v = np.where(np.isnan(v), fill_val, v)
            # -----------------------------------------------------

            # --- Extend longitude by 1 for cyclic continuity ---
            if vname == "longitude":
                v = np.concatenate([v, v[:1] + 360.])
            elif vname in ["uo", "vo", "thetao", "so"]:
                v = np.concatenate([v, v[:, :, :, :1]], axis=-1)
            elif vname in ["zos", "mlotst", "siconc", "sithick"]:
                v = np.concatenate([v, v[:, :, :1]], axis=-1)
            # -----------------------------------------------------

            dst_var[:] = v

    def run(self):
        """Process input file"""
        nc1 = self.input_file
        print(f"Reading {nc1}")
        os.makedirs(self.outdir, exist_ok=True)
        nc2 = os.path.join(self.outdir, os.path.basename(nc1))
        print(f"Saving {nc2}")
        with Dataset(nc1, "r") as src_ds, Dataset(nc2, "w") as dst_ds:
            self.copy_extend_netcdf_file(src_ds, dst_ds)


if __name__ == "__main__":
    args = get_parser().parse_args()
    MakeGlorys12Cyclic(**vars(args)).run()

