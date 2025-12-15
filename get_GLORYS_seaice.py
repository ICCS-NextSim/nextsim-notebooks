import copernicusmarine

#import cdsapi
#import os
#import sys
#import tempfile
#from cdo import *
##import cdo

# User modifiable: southern boundary of the region - must be a string
# User modifiable: temporal frequency of the data (in hours)
temporal_frequency = 1

# ==== END USER MODIFIABLE PARAMETERS ====

# Module instances
#cdo = Cdo()
#client = cdsapi.Client()

#sys.exit()

# Loop parameters
#if len(sys.argv) < 3 or len(sys.argv) > 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
#    print('Usage: ' + sys.argv[0] + ' firstYear lastYear')
#    sys.exit(1)


import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  variables=["siconc", "sithick"],
  minimum_longitude=-180,
  maximum_longitude=179.9166717529297,
  minimum_latitude=-90,
  maximum_latitude=-30,
  start_datetime="2024-01-01T00:00:00",
  end_datetime="2024-12-31T00:00:00",
  minimum_depth=0.49402499198913574,
  maximum_depth=0.49402499198913574,
#  user = "rsantana",
#  pwd = "Ai..$"
)

