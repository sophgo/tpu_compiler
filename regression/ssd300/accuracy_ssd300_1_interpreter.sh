#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

export EVAL_FUNC=eval_ssd.py

$EVAL_FUNC \
    --model=ssd300_opt2.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results.json \
    --count=$1

$EVAL_FUNC \
    --model=ssd300_quant_int8_per_layer.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results.json \
    --count=$1   

$EVAL_FUNC \
    --model=ssd300_quant_int8_per_channel.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results.json \
    --count=$1       

$EVAL_FUNC \
    --model=ssd300_quant_int8_multiplier.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results.json \
    --count=$1    

