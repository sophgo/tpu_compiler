#!/bin/bash
set -e

EVAL_CAFFE_FUNC=eval_caffe_detector_yolo.py

$EVAL_CAFFE_FUNC \
    --model_def $MODEL_DEF \
    --pretrained_model $MODEL_DAT \
    --net_input_dims ${NET_INPUT_DIMS} \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

EVAL_FUNC=eval_yolo.py

$EVAL_FUNC \
    --model=yolo_v3_416.mlir \
    --net_input_dims ${NET_INPUT_DIMS} \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then
  $EVAL_FUNC \
      --model=${NET}_quant_int8_per_tensor.mlir \
      --net_input_dims ${NET_INPUT_DIMS} \
      --obj_threshold 0.005 \
      --nms_threshold 0.45 \
      --dataset=$DATASET_PATH/coco/val2017 \
      --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --result_json=result_416.json \
      --count=$1
fi

if [ $DO_QUANT_INT8_RFHIFT_ONLY -eq 1 ]; then
  $EVAL_FUNC \
      --model=${NET}_quant_int8_rshift_only.mlir \
      --net_input_dims ${NET_INPUT_DIMS} \
      --obj_threshold 0.005 \
      --nms_threshold 0.45 \
      --dataset=$DATASET_PATH/coco/val2017 \
      --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --result_json=result_416.json \
      --count=$1
fi

if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then
  $EVAL_FUNC \
      --model=${NET}_quant_int8_multiplier.mlir \
      --net_input_dims ${NET_INPUT_DIMS} \
      --obj_threshold 0.005 \
      --nms_threshold 0.45 \
      --dataset=$DATASET_PATH/coco/val2017 \
      --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --result_json=result_416.json \
      --count=$1
fi

if [ $DO_QUANT_BF16 -eq 1 ]; then
  $EVAL_FUNC \
      --model=${NET}_quant_bf16.mlir \
      --net_input_dims ${NET_INPUT_DIMS} \
      --obj_threshold 0.005 \
      --nms_threshold 0.45 \
      --dataset=$DATASET_PATH/coco/val2017 \
      --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
      --result_json=result_416.json \
      --count=$1
fi

echo $0 DONE
