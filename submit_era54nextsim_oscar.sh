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

#source /oscar/data/deeps/private/chorvat/nextsim/ERA5/era5.venv/bin/activate

#cd /oscar/data/deeps/private/chorvat/nextsim/ERA5

#year=$((SLURM_ARRAY_TASK_ID+2000))
#script="getERA5_"$year".py"
python era54nextsim.py
#echo "Running job array number: "$SLURM_ARRAY_TASK_ID "input " $input
#echo $script
