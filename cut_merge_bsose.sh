#!/bin/bash


 # extracting layer
 # module load NCO
 ncea -O -F -d Z,5 Uvel_bsoseI139_2013to2021_5dy.nc Uvel_bsoseI139_2013to2021_5dy.nc_26m
 ncea -O -F -d Z,5 Vvel_bsoseI139_2013to2021_5dy.nc Vvel_bsoseI139_2013to2021_5dy.nc_26m
 ncea -O -F -d Z,1 Theta_bsoseI139_2013to2021_5dy.nc Theta_bsoseI139_2013to2021_5dy.nc_2m
 ncea -O -F -d Z,1 Salt_bsoseI139_2013to2021_5dy.nc Salt_bsoseI139_2013to2021_5dy.nc_2m

