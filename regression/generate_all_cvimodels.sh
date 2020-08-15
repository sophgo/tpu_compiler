#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

all_net_list=()

if [ -z $model_list_file ]; then
  model_list_file=$DIR/generic/model_list.txt
fi
while read net bs1 bs4 acc bs1_ext bs4_ext acc_ext
do
  [[ $net =~ ^#.* ]] && continue
  all_net_list+=($net)
done < ${model_list_file}

if [ ! -e cvimodel_release ]; then
  mkdir cvimodel_release
fi

err=0
pushd cvimodel_release
rm -rf working
mkdir working

for net in ${all_net_list[@]}
do
  echo "generate cvimodel for $net"
  pushd working
  NET=$net
  source $DIR/generic/generic_models.sh
  echo "NET=$NET MODEL_TYPE=$MODEL_TYPE"
  if [ -z $MODEL_DAT ]; then
    MODEL_DAT="-"
  fi
  $DIR/convert_model.sh \
      -i ${MODEL_DEF} \
      -d ${MODEL_DAT} \
      -t ${MODEL_TYPE} \
      -b 1 \
      -q ${CALI_TABLE} \
      -l ${USE_LAYERGROUP} \
      -o ${NET}.cvimodel
  mv ${NET}.cvimodel ..
  # generate with detection version if DO_FUSED_POSTPROCESS is set
  if [ $DO_FUSED_POSTPROCESS = "1" ]; then
    $DIR/convert_model.sh \
        -i ${MODEL_DEF_FUSED_POSTPROCESS} \
        -d ${MODEL_DAT} \
        -t ${MODEL_TYPE} \
        -b 1 \
        -q ${CALI_TABLE} \
        -l ${USE_LAYERGROUP} \
        -o ${NET}_with_detection.cvimodel
    mv ${NET}_with_detection.cvimodel ..
  fi
  rm -f ./*
  popd
  if [ "$err" -ne 0 ]; then
    rm -rf working
    popd
    exit 1
  fi
done

rm -rf working
popd
