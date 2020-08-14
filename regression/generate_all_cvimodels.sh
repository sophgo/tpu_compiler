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
  if [ $MODEL_TYPE = "caffe" ]; then
    if [ $USE_LAYERGROUP = "1" ]; then
      $DIR/convert_model_caffe_lg.sh \
        ${MODEL_DEF} \
        ${MODEL_DAT} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
      mv ${NET}.cvimodel ..
    else
      $DIR/convert_model_caffe_df.sh \
        ${MODEL_DEF} \
        ${MODEL_DAT} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
      mv ${NET}.cvimodel ..
    fi
    # generate with detection version if DO_FUSED_POSTPROCESS is set
    if [ $DO_FUSED_POSTPROCESS = "1" ]; then
      if [ $USE_LAYERGROUP = "1" ]; then
        $DIR/convert_model_caffe_lg.sh \
          ${MODEL_DEF_FUSED_POSTPROCESS} \
          ${MODEL_DAT} \
          ${NET} \
          1 \
          ${CALI_TABLE} \
          ${NET}_with_detection.cvimodel
        mv ${NET}_with_detection.cvimodel ..
      else
        $DIR/convert_model_caffe_df.sh \
          ${MODEL_DEF_FUSED_POSTPROCESS} \
          ${MODEL_DAT} \
          ${NET} \
          1 \
          ${CALI_TABLE} \
          ${NET}_with_detection.cvimodel
        mv ${NET}_with_detection.cvimodel ..
      fi
    fi
  elif [ $MODEL_TYPE = "onnx" ]; then
    if [ $USE_LAYERGROUP = "1" ]; then
      $DIR/convert_model_onnx_lg.sh \
        ${MODEL_DEF} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
    else
      $DIR/convert_model_onnx_df.sh \
        ${MODEL_DEF} \
        ${NET} \
        1 \
        ${CALI_TABLE} \
        ${NET}.cvimodel
    fi
    mv ${NET}.cvimodel ..
  elif [ $MODEL_TYPE = "tensorflow" ]; then
    echo "Not supported MODEL_TYPE=$MODEL_TYPE"
  else
    echo "Invalid MODEL_TYPE=$MODEL_TYPE"
    err=1
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
