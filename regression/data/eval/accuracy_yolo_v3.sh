#!/bin/bash
set -e

MLIR_FILES=()
MLIR_TYPES=()
MODEL_DO_PREPROCESS=0
EVAL_CAFFE_FUNC=eval_caffe_detector_yolo.py


if [ -z $YOLO_V3 ]; then
  YOLO_V3=0
fi

if [ -z $YOLO_V4 ]; then
  YOLO_V4=0
fi

if [ -z $SPP_NET ]; then
  SPP_NET=0
fi

if [ -z $TINY ]; then
  TINY=0
fi

$EVAL_CAFFE_FUNC \
    --model_def $MODEL_DEF \
    --pretrained_model $MODEL_DAT \
    --net_input_dims ${NET_INPUT_DIMS} \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --spp_net=$SPP_NET \
    --yolov3 $YOLO_V3 \
    --yolov4 $YOLO_V4 \
    --spp_net=$SPP_NET \
    --tiny=$TINY \
    --count=$1

EVAL_FUNC=eval_yolo.py

if [ $DO_ACCURACY_FP32_INTERPRETER -eq 1 ]; then
  MLIR_FILES+=(${NET}_fp32.mlir)
  MLIR_TYPES+=("fp32")
fi

if [ $DO_QUANT_BF16 -eq 1 ]; then
  MLIR_FILES+=(${NET}_quant_bf16.mlir)
  MLIR_TYPES+=("bf16")
fi

if [ $DO_QUANT_INT8 -eq 1 ]; then
  MLIR_FILES+=(${NET}_quant_int8.mlir)
  MLIR_TYPES+=("int8")
fi

for ((i=0; i<${#MLIR_FILES[@]}; i++))
do
  echo "Eval ${MLIR_TYPES[i]} with interpreter"
  $EVAL_FUNC \
      --model=${MLIR_FILES[i]} \
      --net_input_dims ${NET_INPUT_DIMS} \
      --obj_threshold 0.005 \
      --nms_threshold 0.45 \
      --dataset=$DATASET_PATH/coco/val2017 \
      --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --result_json=result_416.json \
      --model_do_preprocess=${MODEL_DO_PREPROCESS} \
      --spp_net=$SPP_NET \
      --yolov3 $YOLO_V3 \
      --yolov4 $YOLO_V4 \
      --spp_net=$SPP_NET \
      --tiny=$TINY \
      --count=$1
done
echo $0 DONE
