# Script to download wind data from CCMP

#set -x
export LANG=en_US.UTF-8
export HDF5_DISABLE_VERSION_CHECK=1

############# Parametros de entrada ##########
inicio=$1  #{YYYYMMDD}
limite=$2  #{YYYYMMDD}  

# prefixo do arquivo - delayed time or nrt
#NAME=dt_upd_global_merged_msla_h_

PREFIX="ice_conc_sh_polstere-100_multi_" # ice_conc_sh_polstere-100_multi_201901011200.nc
PREFIX="ice_conc_sh_ease-125_multi_" # ice_conc_sh_polstere-100_multi_201901011200.nc
PREFIX="ice_conc_sh_ease2-250_icdr-v2p0_" # 201801081200.nc

##############################################
# WINDS FROM CMEMS
#http://resources.marine.copernicus.eu/documents/PUM/CMEMS-WIND-PUM-012-006.pdf # documentation
#ftp://my.cmems-du.eu/Core/WIND_GLO_WIND_L4_REP_OBSERVATIONS_012_006/CERSAT-GLO-BLENDED_WIND_L4_REP-V6-OBS_FULL_TIME_SERIE/2015/05/
#MYUSER=rsantana
#PASS=RafaelCMEMS2016

current=${inicio}

MYUSER=rafacsantana@gmail.com
PASS=rafacsantana@gmail.com

PATH_OUT=/Users/rsan613/n/southern/data/sic_osisaf/
#PATH_OUT=/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/sic_osisaf/
#PATH_OUT=/Volumes/LaCie/mahuika/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/sic_osisaf/

while [ ${current} -le ${limite} ]; do

  YEAR=`date -u -d "${current}" +%Y`
  MONTH=`date -u -d "${current}" +%m`
  mkdir -p $PATH_OUT/${YEAR}
  echo "Downloading day: ${current}"

  #SOURCE=ftp://ftp.myocean.sltac.cls.fr/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/dataset-duacs-rep-global-merged-allsat-phy-l4-v3/${YEAR}/${PREFIX}_${current}*
  #SOURCE=ftp://ftp.sltac.cls.fr/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/dataset-duacs-rep-global-merged-allsat-phy-l4-v3/${YEAR}/${PREFIX}_${current}*
  #SOURCE=ftp://ftp.remss.com/ccmp/v02.0/Y${YEAR}/M${MONTH}/${PREFIX}_${current}"_V02.0_L3.0_RSS.nc"
  #SOURCE=ftp://ftp.sltac.cls.fr/Core/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/dataset-duacs-nrt-global-merged-allsat-phy-l4-v3/${PREFIX}_${current}*

  #SOURCE=ftp://osisaf.met.no/archive/ice/conc/${YEAR}/${MONTH}/${PREFIX}${current}"1200.nc"
  SOURCE=ftp://osisaf.met.no/reprocessed/ice/conc-cont-reproc/v2p0/${YEAR}/${MONTH}/${PREFIX}${current}"1200.nc"

  echo "${SOURCE}"
  #OUTPUT="${PREFIX}_${current}.nc.gz"
  OUTPUT="${PREFIX}${current}.nc"

  #wget -q -c --retr-symlinks --user=${MYUSER} --password=${PASS} ${SOURCE} -O ${OUTPUT}
  #wget --user=${MYUSER} --password=${PASS} ${SOURCE} 
  wget --ftp-user=anonymous ${SOURCE} -O ${OUTPUT}

  #exit 0
  wait

  #NC=$(basename ${OUTPUT} .gz)
  #gunzip ${OUTPUT}
  wait
  #./cut_adt_aviso.py ${NC} adt/${YEAR}/${OUT_PREFIX}_${current}.nc
  #wait
  #mv ${NC} /storage/remo/data/aviso/${YEAR}/.  ## ATLANTICO

  mv ${OUTPUT} $PATH_OUT/${YEAR}/. 

  current=`date -u -d "${current} +1 day" +%Y%m%d`

done


