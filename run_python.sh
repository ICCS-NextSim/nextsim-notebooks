#!/bin/bash
#SBATCH --job-name=python
#SBATCH --time=24:00:00
#SBATCH --nodes=1   
#SBATCH --mem-per-cpu=1G 
#SBATCH --ntasks=40
#SBATCH --output=log_python.log
#SBATCH -A uoa03669
#SBATCH --partition=milan

echo Remember to load python for nextsim
echo 'load-python-nextsim'

echo Running ${1}

python ${1} # valid_nextsim.py_sub2


