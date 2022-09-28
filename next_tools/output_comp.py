"""
This module contains tests to check that NextSim output is valid.
"""

import numpy as np
import xarray as xr

def output_comp(file1, file2, eps=1e-2):
    """
    Load a pair of netcdf model outputs and run a sequence
    of comparison tests.

    Parameters
    ----------
    file1 : str
        Filename of first model output
    file2 : str
        Filename of second model output
    eps : float
        Maximum relative error
    """

    data1 = xr.open_dataset(file1)
    data2 = xr.open_dataset(file2)

    fields = ["sit", "sic"]

    for i in range(len(fields)):
        assert np.nanmax(getattr(data1, fields[i]).data
                         - getattr(data2, fields[i]).data
                         ) < eps * np.nanmax(getattr(data1, fields[i]).data), fields[i] + "difference too large."
