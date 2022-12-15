#!/bin/bash

# Script to download BSOSE

#curl -C - -O http://sose.ucsd.edu/SO6/ITER139/Vvel_bsoseI139_2019to2021_GVAL_5dy.nc

if [ $1 == 'ocean' ] 
then

  echo Downloading ocean var BSOSE between 2019 and 2021
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/Uvel_bsoseI139_2019to2021_GVAL_5dy.nc
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/Vvel_bsoseI139_2019to2021_GVAL_5dy.nc
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/SSH_bsoseI139_2019to2021_GVAL_5dy.nc
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/Salt_bsoseI139_2019to2021_GVAL_5dy.nc
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/Theta_bsoseI139_2019to2021_GVAL_5dy.nc
       #-C - -O http://sose.ucsd.edu/SO6/ITER139/Strat_bsoseI139_2019to2021_GVAL_5dy.nc

elif [ $1 == 'ice' ] 
then

  echo Downloading ice var BSOSE between 2019 and 2021
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/SeaIceArea_bsoseI139_2013to2021_5dy.nc
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/SeaIceHeff_bsoseI139_2013to2021_5dy.nc
       
elif [ $1 == 'mld' ] 
then

  echo Downloading MLD BSOSE between 2013 and 2021
  curl -C - -O http://sose.ucsd.edu/SO6/ITER139/MLD_bsoseI139_2013to2021_5dy.nc

elif [ $1 == '2013' ] 
then

  echo Downloading BSOSE between 2013 and 2021
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Uvel_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Vvel_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Salt_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/Theta_bsoseI139_2013to2021_5dy.nc
  curl -C - -O  http://sose.ucsd.edu/SO6/ITER139/SSH_bsoseI139_2013to2021_5dy.nc

fi


