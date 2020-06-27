#!/bin/bash
set -e

# with data_quant
model_list_1=(
  "resnet50"
  "mobilenet_v1"
  "mobilenet_v2"
  "googlenet"
  "squeezenet"
  "inception_v3"
  "inception_v4"
  "vgg16"
  "shufflenet_v2"
  "densenet_121"
  "densenet_201"
  "senet_res50"
  "arcface_res50"
  "retinaface_mnet25"
  "retinaface_mnet25_600"
  "retinaface_res50"
  "ssd300"
  "yolo_v2_416"
  "yolo_v2_1080"
  "yolo_v3_320"
  "yolo_v3_416"
  "yolo_v3_608"
  # "resnet18"
  # "efficientnet_b0"
  # "alphapose"
  # "espcn_3x"
  # "unet"
)

# with input_data
model_list_2=(
  "resnet18"
  "efficientnet_b0"
  "alphapose"
  "espcn_3x"
  "unet"
)

for model in ${model_list_1[@]}
do
  echo "extract $model"
  cvi_npz_tool.py extract \
      ${model}_out_all.npz \
      ${model}_in_int8.npz \
      data_quant
done

for model in ${model_list_2[@]}
do
  echo "extract $model"
  cvi_npz_tool.py extract \
      ${model}_out_all.npz \
      ${model}_in_int8.npz \
      input_quant
done
