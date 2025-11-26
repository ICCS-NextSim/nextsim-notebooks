#!/bin/bash

# Resource request
#SBATCH --time=8:00:00
####SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G

# Partition
######SBATCH --partition=batch

# Provide index values (TASK IDs)
#####SBATCH --array=13

# Job handling
#SBATCH -J era54nextsim
#SBATCH -o log_era54nextsim.out
#####SBATCH -o %x-%A_%a.out

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
#module load hpcx-mpi
#module load cdo-mpi

echo "FIRST RUN: mamba activate /oscar/data/deeps/private/chorvat/data/ERA5/era5-postproc"

dir_in="/oscar/data/deeps/private/chorvat/data/GLORYS/"

dir_out="/oscar/data/deeps/private/chorvat/data/GLORYS/proc/"

prefix="cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-30.00S_1.54m_"
#prefix="cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo-zos_180.00W-179.92E_80.00S-30.00S_29.44m_"

for year in {2014..2024}; do
  yearp1=$((year+1))

  file_in=${dir_in}${prefix}${year}"-01-01-"${yearp1}"-01-01.nc"

  echo $file_in

  python glorys4nextsim.py $file_in $dir_out

done

#echo "Running job array number: "$SLURM_ARRAY_TASK_ID "input " $input
#echo $script

