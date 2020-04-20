#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

net_list=(
  "bmface_v3"
  # "liveness"
)

if [ ! -e cvimodel_release ]; then
  mkdir cvimodel_release
fi

pushd cvimodel_release
rm -rf working
mkdir working

# generic
for net in ${net_list[@]}
do
  echo "generate cvimodel for $net"
  pushd working
  NET=$net
  source $DIR/cvitek_zoo_models.sh
  if [ $MODEL_TYPE = "caffe" ]; then
    $REGRESSION_PATH/convert_model_caffe.sh \
      ${MODEL_DEF} \
      ${MODEL_DAT} \
      1 \
      ${CALI_TABLE} \
      ${NET}.cvimodel
  elif [ $MODEL_TYPE = "onnx" ]; then
    $REGRESSION_PATH/convert_model_onnx.sh \
      ${MODEL_DEF} \
      ${NET} \
      1 \
      ${CALI_TABLE} \
      ${NET}.cvimodel
  else
    echo "Invalid MODEL_TYPE=$MODEL_TYPE"
    return 1
  fi
  mv ${NET}.cvimodel ..
  rm ./*
  popd
done

rm -rf working
popd
