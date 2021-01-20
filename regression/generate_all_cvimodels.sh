#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

WORKING_PATH=${WORKING_PATH:-$DIR}
mkdir -p $WORKING_PATH/cvimodel_release

all_net_list=()

if [ -z $model_list_file ]; then
  model_list_file=$DIR/generic/model_list.txt
fi
while read net bs1 bs4 acc bs1_ext bs4_ext acc_ext
do
  [[ $net =~ ^#.* ]] && continue
  all_net_list+=($net)
done < ${model_list_file}


err=0
pushd $WORKING_PATH/cvimodel_release
rm -rf working
mkdir -p working

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
  if [[ "$MODEL_TYPE" == "tflite_int8" ]]; then
    echo "Not Generate $MODEL_TYPE model to cvimodel, only regression test"
    popd
    continue
  fi
  $DIR/convert_model.sh \
      -i ${MODEL_DEF} \
      -d ${MODEL_DAT} \
      -t ${MODEL_TYPE} \
      -b 1 \
      -q ${CALI_TABLE} \
      -l ${USE_LAYERGROUP} \
      -v ${SET_CHIP_NAME} \
      -z ${NET_INPUT_DIMS} \
      -y ${IMAGE_RESIZE_DIMS} \
      -r ${RAW_SCALE} \
      -m ${MEAN} \
      -s ${STD} \
      -a ${INPUT_SCALE} \
      -w ${MODEL_CHANNEL_ORDER} \
      -o ${NET}.cvimodel
  mv ${NET}.cvimodel ..
  # generate with detection version if DO_FUSED_POSTPROCESS is set
  if [[ $DO_FUSED_POSTPROCESS = "1" ]]; then
    $DIR/convert_model.sh \
        -i ${MODEL_DEF_FUSED_POSTPROCESS} \
        -d ${MODEL_DAT} \
        -t ${MODEL_TYPE} \
        -b 1 \
        -q ${CALI_TABLE} \
        -l ${USE_LAYERGROUP} \
        -v ${SET_CHIP_NAME} \
        -z ${NET_INPUT_DIMS} \
        -y ${IMAGE_RESIZE_DIMS} \
        -r ${RAW_SCALE} \
        -m ${MEAN} \
        -s ${STD} \
        -a ${INPUT_SCALE} \
        -w ${MODEL_CHANNEL_ORDER} \
        -o ${NET}_with_detection.cvimodel
    mv ${NET}_with_detection.cvimodel ..
  fi
  # generate fuse_preprocess version if DO_FUSED_PREPROCESS is set
  if [[ $DO_FUSED_PREPROCESS == "1" ]]; then
    $DIR/convert_model.sh \
        -i ${MODEL_DEF} \
        -d ${MODEL_DAT} \
        -t ${MODEL_TYPE} \
        -b 1 \
        -q ${CALI_TABLE} \
        -l ${USE_LAYERGROUP} \
        -v ${SET_CHIP_NAME} \
        -p \
        -z ${NET_INPUT_DIMS} \
        -y ${IMAGE_RESIZE_DIMS} \
        -r ${RAW_SCALE} \
        -m ${MEAN} \
        -s ${STD} \
        -a ${INPUT_SCALE} \
        -w ${MODEL_CHANNEL_ORDER} \
        -o ${NET}_fused_preprocess.cvimodel
    mv ${NET}_fused_preprocess.cvimodel ..
  fi
  # for both DO_FUSED_PREPROCESS and DO_FUSED_POSTPROCESS are set
  if [[ $DO_FUSED_PREPROCESS == "1" && $DO_FUSED_POSTPROCESS = "1" ]]; then
    $DIR/convert_model.sh \
        -i ${MODEL_DEF_FUSED_POSTPROCESS} \
        -d ${MODEL_DAT} \
        -t ${MODEL_TYPE} \
        -b 1 \
        -q ${CALI_TABLE} \
        -l ${USE_LAYERGROUP} \
        -v ${SET_CHIP_NAME} \
        -p \
        -z ${NET_INPUT_DIMS} \
        -y ${IMAGE_RESIZE_DIMS} \
        -r ${RAW_SCALE} \
        -m ${MEAN} \
        -s ${STD} \
        -a ${INPUT_SCALE} \
        -w ${MODEL_CHANNEL_ORDER} \
        -o ${NET}_fused_preprocess_with_detection.cvimodel
    mv ${NET}_fused_preprocess_with_detection.cvimodel ..
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
