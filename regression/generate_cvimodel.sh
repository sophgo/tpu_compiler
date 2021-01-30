#!/bin/bash
set -e

NET=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

TMP_PATH=working_$NET
mkdir -p $TMP_PATH
pushd $TMP_PATH

source $DIR/generic/generic_models.sh
echo "generate cvimodel for $NET"
echo "NET=$NET MODEL_TYPE=$MODEL_TYPE"

if [ -z $MODEL_DAT ]; then
  MODEL_DAT="-"
fi
if [[ "$MODEL_TYPE" == "tflite_int8" ]]; then
  echo "Not Generate $MODEL_TYPE model to cvimodel, only regression test"
  popd

  rm -rf $TMP_PATH
  unset NET
  exit
fi

$DIR/convert_model.sh \
    -i ${MODEL_DEF} \
    -d ${MODEL_DAT} \
    -t ${MODEL_TYPE} \
    -b 1 \
    -q ${CALI_TABLE} \
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

popd

rm -rf $TMP_PATH

unset NET
