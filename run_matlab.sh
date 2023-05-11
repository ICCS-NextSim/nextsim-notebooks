#!/bin/bash
#SBATCH --job-name=python
#SBATCH --time=24:00:00
#SBATCH --nodes=1   
#SBATCH --mem-per-cpu=1G 
#SBATCH --ntasks=32
#SBATCH --output=log_matlab.log
#SBATCH -A uoa03669
#SBATCH --partition=milan

echo Remember to load python for nextsim
echo 'load-matlab'

matlab -nodisplay -nosplash -nodesktop -r "run('analyse_nextsim_sub.m');exit;"
#python valid_nextsim.py_sub


