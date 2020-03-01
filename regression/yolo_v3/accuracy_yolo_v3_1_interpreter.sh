#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=eval_yolo.py

$EVAL_FUNC \
    --model=yolo_v3_416.mlir \
    --net_input_dims 416,416 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

$EVAL_FUNC \
    --model=yolo_v3_416_quant_int8_per_layer.mlir \
    --net_input_dims 416,416 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

$EVAL_FUNC \
    --model=yolo_v3_416_quant_int8_per_channel.mlir \
    --net_input_dims 416,416 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

$EVAL_FUNC \
    --model=yolo_v3_416_quant_int8_multiplier.mlir \
    --net_input_dims 416,416 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

$EVAL_FUNC \
    --model=yolo_v3_416_quant_bf16.mlir \
    --net_input_dims 416,416 \
    --obj_threshold 0.005 \
    --nms_threshold 0.45 \
    --dataset=$DATASET_PATH/coco/val2017 \
    --annotations=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --result_json=result_416.json \
    --count=$1

echo $0 DONE
