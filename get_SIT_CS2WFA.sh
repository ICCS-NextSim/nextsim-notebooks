# Script to download wind data from CCMP

#set -x
export LANG=en_US.UTF-8
export HDF5_DISABLE_VERSION_CHECK=1

############# Parametros de entrada ##########
inicio=$1  #{YYYYMMDD}
limite=$2  #{YYYYMMDD}  

# prefixo do arquivo - delayed time or nrt
#NAME=dt_upd_global_merged_msla_h_

PREFIX="CS2WFA_25km_" # seaice_conc_daily_sh_20220101_f17_v04r00.nc

##############################################
# WINDS FROM CMEMS
#http://resources.marine.copernicus.eu/documents/PUM/CMEMS-WIND-PUM-012-006.pdf # documentation
#ftp://my.cmems-du.eu/Core/WIND_GLO_WIND_L4_REP_OBSERVATIONS_012_006/CERSAT-GLO-BLENDED_WIND_L4_REP-V6-OBS_FULL_TIME_SERIE/2015/05/
#MYUSER=rsantana
#PASS=RafaelCMEMS2016

current=${inicio}

MYUSER=rafacsantana@gmail.com
PASS=rafacsantana@gmail.com

PATH_OUT=/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/sit_cs2wfa/

while [ ${current} -le ${limite} ]; do

  YEAR=`date -u -d "${current}" +%Y`
  MONTH=`date -u -d "${current}" +%m`
  mkdir -p $PATH_OUT/${YEAR}
  echo "Downloading day: ${current}"

  #SOURCE=ftp://ftp.myocean.sltac.cls.fr/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/dataset-duacs-rep-global-merged-allsat-phy-l4-v3/${YEAR}/${PREFIX}_${current}*
  #SOURCE=ftp://ftp.sltac.cls.fr/Core/SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047/dataset-duacs-rep-global-merged-allsat-phy-l4-v3/${YEAR}/${PREFIX}_${current}*
  #SOURCE=ftp://ftp.remss.com/ccmp/v02.0/Y${YEAR}/M${MONTH}/${PREFIX}_${current}"_V02.0_L3.0_RSS.nc"
  #SOURCE=ftp://ftp.sltac.cls.fr/Core/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/dataset-duacs-nrt-global-merged-allsat-phy-l4-v3/${PREFIX}_${current}*

  SOURCE=https://zenodo.org/record/7327711/files/${PREFIX}${YEAR}${MONTH}".nc"

  echo "${SOURCE}"
  #OUTPUT="${PREFIX}_${current}.nc.gz"
  OUTPUT=${PREFIX}${YEAR}${MONTH}".nc"

  #wget -q -c --retr-symlinks --user=${MYUSER} --password=${PASS} ${SOURCE} -O ${OUTPUT}
  #wget --user=${MYUSER} --password=${PASS} ${SOURCE} 
  wget ${SOURCE} #-O ${OUTPUT}

  #exit 0
  wait

  #NC=$(basename ${OUTPUT} .gz)
  #gunzip ${OUTPUT}
  wait
  #./cut_adt_aviso.py ${NC} adt/${YEAR}/${OUT_PREFIX}_${current}.nc
  #wait
  #mv ${NC} /storage/remo/data/aviso/${YEAR}/.  ## ATLANTICO

  mv ${OUTPUT} $PATH_OUT/${YEAR}/. 

  current=`date -u -d "${current} +1 month" +%Y%m%d`

done


