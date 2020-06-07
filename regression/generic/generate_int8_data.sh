#!/bin/bash
set -e

# with data_quant
model_list_1=(
  "resnet50"
  "vgg16"
  "mobilenet_v1"
  "mobilenet_v2"
  "googlenet"
  "inception_v3"
  "inception_v4"
  "squeezenet"
  "shufflenet_v2"
  "densenet_121"
  "densenet_201"
  "senet_res50"
  "arcface_res50"
  "retinaface_mnet25"
  "retinaface_res50"
  "ssd300"
  "yolo_v2_1080"
  "yolo_v2_416"
  "yolo_v3_608"
  "yolo_v3_416"
  "yolo_v3_320"
  # "resnet18"
  # "efficientnet_b0"
  # "alphapose"
)

# with input_data
model_list_2=(
  "resnet18"
  "efficientnet_b0"
  "alphapose"
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

