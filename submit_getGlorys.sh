#!/bin/bash
# Resource request
#SBATCH --time=8:00:00
####SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60G

# Partition
######SBATCH --partition=batch

# Provide index values (TASK IDs)
#####SBATCH --array=13

# Job handling
#SBATCH -J glorys
#SBATCH -o log_glorys.out
#####SBATCH -o %x-%A_%a.out

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
#module load hpcx-mpi
#module load cdo-mpi

#mamba activate /oscar/data/deeps/private/chorvat/data/GLORYS/glorys

#year=$((SLURM_ARRAY_TASK_ID+2000))
#script="getERA5_"$year".py"

python get_GLORYS_3m.py
#python get_GLORYS_30m.py
#python get_GLORYS_seaice.py
#python get_GLORYS_ice_drift.py

#echo "Running job array number: "$SLURM_ARRAY_TASK_ID "input " $input
#echo $script
