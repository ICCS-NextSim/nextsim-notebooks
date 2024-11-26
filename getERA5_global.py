#!/scale_wlg_persistent/filesets/project/uoa03669/rsan613/conda/envs/era5/bin/python


# Script to download ERA5 solutions and processing it for neXtSIM friendly.
# See https://cds.climate.copernicus.eu/api-how-to for the neccesary requirements.
# NB! I had to do pip install requests, in addition to pip install cdsapi
# run: conda install -c conda-forge cdo python-cdo # rafa 17/09/2022


import cdsapi
import os
import sys
import tempfile
from cdo import *

# User modifiable: southern boundary of the region - must be a string
nlat='90'
slat='-90'
# User modifiable: temporal frequency of the data (in hours)
temporal_frequency = 1

# ==== END USER MODIFIABLE PARAMETERS ====

# Module instances
cdo = Cdo()
client = cdsapi.Client()

#sys.exit()

# Loop parameters
#if len(sys.argv) < 3 or len(sys.argv) > 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
#    print('Usage: ' + sys.argv[0] + ' firstYear lastYear')
#    sys.exit(1)

firstYear = 2020 # int(sys.argv[1])
lastYear  = 2020 # int(sys.argv[2])

# Request parameters
product = 'reanalysis-era5-single-levels'

variable =  ['10m_u_component_of_wind', 
             '10m_v_component_of_wind']

#            'surface_solar_radiation_downwards',   # not necessary
#            'surface_thermal_radiation_downwards'] # not necessary

product_type = 'reanalysis'
filename = 'ERA5_{}_y{:4d}.nc'
format = 'netcdf'

time = []
for h in range(0,24,temporal_frequency):
    time.append('{0:02d}'.format(h) + ':00')

day = []
for d in range(1,32):
    day.append('{0:02d}'.format(d))

month = []
for m in range(1,13):
    month.append('{0:02d}'.format(m))

# Say what we'll do
print('Will fetch ERA5 '+product_type+' from start of '+str(firstYear)+' to end of '+str(lastYear))
print('Temporal frequency set at '+str(temporal_frequency)+' hours')
print('Southern boundary set at '+slat)

########################################################################
# Loop over years and months
########################################################################
            
for year in range(firstYear,lastYear+1):
    for var in variable:

        print('\nRequesting variable '+var+' for '+str(year)+'\n')

        result = client.retrieve( product,
            {'variable':var,
             'product_type':product_type,
             'year':year,
             'month':month,
             'day':day,
             'time':time,
             'format':format} )

        # Set $TMPDIR, $TEMP, or $TMP to $PWD to write the temporary file in
        # the current directory
        f = tempfile.NamedTemporaryFile(delete=False)
        temp_file = f.name
        f.close()
        result.download(temp_file)

        # Use cdo to select everything north of slat
        short_name = cdo.showname(input=temp_file)
        cdo.sellonlatbox('0,360,'+slat+','+nlat,
            input=temp_file, output=filename.format(short_name[0],year))
        
        # Remove the temp_file
        os.unlink(temp_file)

        print('\nOutput written to '+filename.format(short_name[0],year)+'\n')

