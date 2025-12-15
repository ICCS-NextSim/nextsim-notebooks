import copernicusmarine

#import cdsapi
#import os
import sys
#import tempfile
#from cdo import *
##import cdo

# User modifiable: southern boundary of the region - must be a string
nlat=-38
slat=-81
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

#firstYear = 2013 # int(sys.argv[1])
#lastYear  = 2013 # int(sys.argv[2])

#copernicusmarine.subset(
#  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
#  dataset_version="202311",
#  variables=["siconc", "mlotst", "so", "thetao", "sithick"],
#  minimum_longitude=-180,
#  maximum_longitude=179.9166717529297,
#  minimum_latitude=-80,
#  maximum_latitude=-38,
#  start_datetime="2013-01-01T00:00:00",
#  end_datetime="2014-01-01T00:00:00",
#  minimum_depth=0.49402499198913574,
#  maximum_depth=0.49402499198913574,
#  coordinates_selection_method="strict-inside",
#  netcdf_compression_level=1,
#  disable_progress_bar=True,
#)



copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  dataset_version="202311",
  variables=["uo", "vo", "zos"],
  minimum_longitude=-180,
  maximum_longitude=179.9166717529297,
  minimum_latitude=-80,
  maximum_latitude=-30,
  start_datetime="2015-01-01T00:00:00",
  end_datetime="2015-12-31T00:00:00",
  minimum_depth=29.444730758666992,
  maximum_depth=29.444730758666992,
  coordinates_selection_method="strict-inside",
  netcdf_compression_level=1,
  disable_progress_bar=True,
)



#  user = "rsantana",
#  pwd = "A....$"


## Request parameters
#product = 'reanalysis-era5-single-levels'
#
##variable = ['2m_dewpoint_temperature',
##            '2m_temperature',
##            'mean_sea_level_pressure',
##            'mean_total_precipitation_rate',
##            'mean_surface_downward_short_wave_radiation_flux',
##            'mean_surface_downward_long_wave_radiation_flux',
##            'mean_snowfall_rate',
##            '10m_u_component_of_wind', 
##            '10m_v_component_of_wind']
#
#variable = ['10m_u_component_of_wind',
#            '10m_v_component_of_wind']
#
##            'surface_solar_radiation_downwards',   # not necessary
##            'surface_thermal_radiation_downwards'] # not necessary
#
#product_type = 'reanalysis'
#filename = 'ERA5_{}_y{:4d}.nc'
#format = 'netcdf'
#
#time = []
#for h in range(0,24,temporal_frequency):
#    time.append('{0:02d}'.format(h) + ':00')
#
#day = []
#for d in range(1,32):
#    day.append('{0:02d}'.format(d))
#
#month = []
#for m in range(1,13):
#    month.append('{0:02d}'.format(m))
#
## Say what we'll do
#print('Will fetch ERA5 '+product_type+' from start of '+str(firstYear)+' to end of '+str(lastYear))
#print('Temporal frequency set at '+str(temporal_frequency)+' hours')
#print('Southern boundary set at '+slat)
#
#########################################################################
## Loop over years and months
#########################################################################
#            
#for year in range(firstYear,lastYear+1):
#    for var in variable:
#
#        print('\nRequesting variable '+var+' for '+str(year)+'\n')
#
#        result = client.retrieve( product,
#            {'variable':var,
#             'product_type':product_type,
#             'year':year,
#             'month':month,
#             'day':day,
#             'time':time,
#             'format':format} )
#
#        # Set $TMPDIR, $TEMP, or $TMP to $PWD to write the temporary file in
#        # the current directory
#        f = tempfile.NamedTemporaryFile(delete=False)
#        temp_file = f.name
#        f.close()
#        result.download(temp_file)
#
#        # Use cdo to select everything north of slat
#        short_name = cdo.showname(input=temp_file)
#        cdo.sellonlatbox('0,360,'+slat+','+nlat,
#            input=temp_file, output=filename.format(short_name[0],year))
#        
#        # Remove the temp_file
#        os.unlink(temp_file)
#
#        print('\nOutput written to '+filename.format(short_name[0],year)+'\n')






