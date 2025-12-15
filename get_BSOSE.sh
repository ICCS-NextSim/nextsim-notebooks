#!/bin/bash

# Script to download BSOSE

#curl -C - -O http://sose.ucsd.edu/SO6/ITER139/Vvel_bsoseI139_2019to2021_GVAL_5dy.nc

cd /oscar/data/deeps/private/chorvat/data/BSOSE/

pwd

#exit 0

if [ $1 == 'ITER139' ]
then

  echo Downloading BSOSE between 2013 and 2021
 # wget -c  http://sose.ucsd.edu/SO6/ITER139/MLD_bsoseI139_2013to2021_5dy.nc
 # wget -c  http://sose.ucsd.edu/SO6/ITER139/Uvel_bsoseI139_2013to2021_5dy.nc
  wget -c  http://sose.ucsd.edu/SO6/ITER139/Vvel_bsoseI139_2013to2021_5dy.nc
 # wget -c  http://sose.ucsd.edu/SO6/ITER139/Salt_bsoseI139_2013to2021_5dy.nc
 # wget -c  http://sose.ucsd.edu/SO6/ITER139/Theta_bsoseI139_2013to2021_5dy.nc
 # wget -c  http://sose.ucsd.edu/SO6/ITER139/SSH_bsoseI139_2013to2021_5dy.nc
 # wget -c https://sose.ucsd.edu/SO6/ITER139/SeaIceArea_bsoseI139_2013to2021_5dy.nc
 #  wget -c https://sose.ucsd.edu/SO6/ITER139/SeaIceHeff_bsoseI139_2013to2021_5dy.nc
elif [ $1 == '2013' ]
then

  echo Downloading BSOSE between 2013 and 2021
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Uvel_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Vvel_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Salt_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Theta_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/SSH_bsoseI139_2013to2021_5dy.nc

fi


