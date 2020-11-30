#!/bin/bash
set -e
# set -o pipefail

model_list_lg=(
  "resnet50"
  "vgg16"
  "mobilenet_v1"
  "mobilenet_v2"
  "googlenet"
  "inception_v3"
  "inception_v4"
  "squeezenet"
  "shufflenet_v2"
  # "densenet_121"
  # "densenet_201"
  # "senet_res50"
  # "arcface_res50"
  "retinaface_mnet25"
  # "retinaface_res50"
  "ssd300"
  # "yolo_v2_1080"
  # "yolo_v2_416"
  # "yolo_v3_608"
  # "yolo_v3_416"
  # "yolo_v3_320"
  # "resnet18"
  "efficientnet_b0"
  "alphapose"
)

model_list_df=(
  # "resnet50"
  # "vgg16"
  # "mobilenet_v1"
  # "mobilenet_v2"
  # "googlenet"
  # "inception_v3"
  # "inception_v4"
  # "squeezenet"
  # "shufflenet_v2"
  "densenet_121"
  "densenet_201"
  "senet_res50"
  "arcface_res50"
  # "retinaface_mnet25"
  "retinaface_res50"
  # "ssd300"
  "yolo_v2_1080"
  "yolo_v2_416"
  "yolo_v3_608"
  "yolo_v3_416"
  "yolo_v3_320"
  "resnet18"
  # "efficientnet_b0"
  # "alphapose"
)

root_dir=$1

# layer_group
for net in ${model_list_lg[@]}
do
  echo "profile model for $net"
  pushd ${root_dir}/${net}_bs1
  tpuc-translate \
    --mlir-to-cmdbuf \
    ${net}_quant_int8_multiplier_layergroup.mlir \
    -o ${net}_cmdbuf.bin
  popd
  rm -rf ${net}
  mkdir -p ${net}
  mv ${root_dir}/${net}_bs1/${net}_cmdbuf.bin ${net}/
  pushd ${net}
  cvi_profiling --cmdbuf ${net}_cmdbuf.bin
  popd
done

# deep fusion
for net in ${model_list_df[@]}
do
  echo "profile model for $net"
  pushd ${root_dir}/${net}_bs1
  tpuc-translate \
    --mlir-to-cmdbuf \
    ${net}_quant_int8_multiplier_tl_lw.mlir \
    -o ${net}_cmdbuf.bin
  popd
  rm -rf ${net}
  mkdir -p ${net}
  mv ${root_dir}/${net}_bs1/${net}_cmdbuf.bin ${net}/
  pushd ${net}
  cvi_profiling --cmdbuf ${net}_cmdbuf.bin
  popd
done

