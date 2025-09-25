#!/bin/bash
#SBATCH --job-name=get_ERA5
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=24G
#SBATCH --ntasks=4
####SBATCH --output=log_nextsim.log
#SBATCH -A uoa03669
#SBATCH --hint=nomultithread

#export TMP=$PWD

#./getERA5_global.py
#python /home/rsan613/scripts/uo/nextsim-notebooks/getERA5.py  > log_getERA5.out
python era54nextsim_oscar.py  > log_era54nextsim.out
