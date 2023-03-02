#!/bin/bash
#SBATCH --job-name=python
#SBATCH --time=24:00:00
#SBATCH --nodes=1   
#SBATCH --mem-per-cpu=1G 
#SBATCH --ntasks=32
#SBATCH --output=log_python.log
#SBATCH -A uoa03669
#SBATCH --partition=milan

echo Remember to load python for nextsim
echo 'load-python-nextsim'

python valid_nextsim.py_sub

#srun singularity exec -B /nesi/nobackup/uoa03669 $NEXTSIM_IMAGE_NAME nextsim.exec --config-files=/config_files/nextsim.cfg

#echo Deleting forcing files from expt dir to save space ...
#rm -rf $NEXTSIM_INPUT_DIR/{ERA*,ETOPO*,GLORYS*} 

