# Script to download wind data from CCMP

#set -x
export LANG=en_US.UTF-8
export HDF5_DISABLE_VERSION_CHECK=1

############# Parametros de entrada ##########
inicio=$1  #{YYYYMMDD}
limite=$2  #{YYYYMMDD}  

# prefixo do arquivo - delayed time or nrt
#NAME=dt_upd_global_merged_msla_h_

PREFIX="ice_drift_sh_ease2-750_cdr-v1p0_24h-" # ice_conc_sh_polstere-100_multi_201901011200.nc

#ftp://osisaf.met.no/reprocessed/ice/drift_lr/v1/merged/2017/01/ice_drift_sh_ease2-750_cdr-v1p0_24h-201701071200.nc

current=${inicio}

MYUSER=rafacsantana@gmail.com
PASS=rafacsantana@gmail.com

PATH_OUT=/Users/rsan613/n/southern/data/drift_osisaf_ease2/
PATH_OUT=/scale_wlg_nobackup/filesets/nobackup/uoa03669/data/drift_osisaf_ease2/
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


  SOURCE=ftp://osisaf.met.no/reprocessed/ice/drift_lr/v1/merged/${YEAR}/${MONTH}/${PREFIX}${current}"1200.nc"

  echo "${SOURCE}"

  OUTPUT=${PREFIX}${current}"1200.nc"

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


