#!/bin/bash
set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

generic_net_list=(
  "resnet50"
  "vgg16"
  "mobilenet_v2"
  "googlenet"
  "inception_v3"
  "inception_v4"
  "shufflenet_v2"
  "squeezenet"
  "arcface_res50"
  "retinaface_mnet25"
  "retinaface_res50"
  "ssd300"
  "yolo_v3_416"
  # "yolo_v3_320"
  # "resnet18"
  "efficientnet_b0"
  "alphapose"
)

extra_net_param()
{
  NET=$1

  if [ $NET = "retinaface_mnet25_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/mnet_320_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_mnet25_calibration_table
  fi

  if [ $NET = "retinaface_res50_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/retinaface_res50_calibration_table
  fi

  if [ $NET = "yolo_v3_416_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
  fi

  if [ $NET = "yolo_v3_320_with_detection" ]; then
  export MODEL_TYPE="caffe"
  export MODEL_DEF=$MODEL_PATH/object_detection/yolo_v3/caffe/yolov3_320_with_detection.prototxt
  export MODEL_DAT=$MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel
  export CALI_TABLE=$REGRESSION_PATH/data/cali_tables/yolo_v3_calibration_table_autotune
  fi
}

extra_net_list=(
  "retinaface_mnet25_with_detection"
  "retinaface_res50_with_detection"
  "yolo_v3_416_with_detection"
  "yolo_v3_320_with_detection"
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
  if [ $MODEL_TYPE = "caffe" ]; then
    $DIR/convert_model_caffe.sh \
      ${MODEL_DEF} \
      ${MODEL_DAT} \
      1 \
      ${CALI_TABLE} \
      ${NET}.cvimodel
  elif [ $MODEL_TYPE = "onnx" ]; then
    $DIR/convert_model_onnx.sh \
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

# extra
for net in ${extra_net_list[@]}
do
  echo "generate cvimodel for $net"
  pushd working
  NET=$net
  extra_net_param $NET
  if [ $MODEL_TYPE = "caffe" ]; then
    $DIR/convert_model_caffe.sh \
      ${MODEL_DEF} \
      ${MODEL_DAT} \
      1 \
      ${CALI_TABLE} \
      ${NET}.cvimodel
  elif [ $MODEL_TYPE = "onnx" ]; then
    $DIR/convert_model_onnx.sh \
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
