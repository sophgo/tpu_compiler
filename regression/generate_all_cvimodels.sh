#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

generic_net_list=(
  "resnet50"
  "vgg16"
  "mobilenet_v2"
  # "googlenet"
  "inception_v3"
  "inception_v4"
  "efficientnet_b0"
  "shufflenet_v2"
  "squeezenet"
  "arcface_res50"
  "bmface_v3"
  # "liveness"
  "retinaface_mnet25"
  "retinaface_res50"
  "ssd300"
  "yolo_v3_416"
  # "yolo_v3_608"
  "yolo_v3_320"
  # "yolo_v3_160"
  # "yolo_v3_512x288"
  "alphapose"
)

get_net_param()
{
  NET=$1

  if [ $NET = "retinaface_mnet25_with_detection" ]; then
  export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
  fi

  if [ $NET = "retinaface_res50_with_detection" ]; then
  export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_res50_calibration_table
  fi
}

extra_net_list=(
  "retinaface_mnet25_with_detection"
  "retinaface_res50_with_detection"
)

if [ ! -e cvimodel_release ]; then
  mkdir cvimodel_release
fi

pushd cvimodel_release
rm -rf working
mkdir working

# generic
for net in ${generic_net_list[@]}
do
  echo "generate cvimodel for $net"
  pushd working
  NET=$net
  source $DIR/generic/generic_models.sh
  $DIR/convert_model_caffe.sh \
    ${MODEL_DEF} \
    ${MODEL_DAT} \
    1 \
    ${CALI_TABLE} \
    ${NET}.cvimodel
  mv ${NET}.cvimodel ..
  rm ./*
  popd
done

# extra
for net in ${extra_net_list[@]}
do
  echo "generate cvimodel for $net"
  pushd working
  NET=$net
  get_net_param $NET
  $DIR/convert_model_caffe.sh \
    ${MODEL_DEF} \
    ${MODEL_DAT} \
    1 \
    ${CALI_TABLE} \
    ${NET}.cvimodel
  mv ${NET}.cvimodel ..
  rm ./*
  popd
done

rm -rf working
popd
