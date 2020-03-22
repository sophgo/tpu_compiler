#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


export EVAL_FUNC=eval_ssd.py

if [[ $2 -eq 1 ]]; then
$EVAL_FUNC \
    --model=ssd300_opt2.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_fp32_interpreter.json \
    --count=$1 \
    --pre_result_json=./coco_results_fp32_interpreter.json
else
$EVAL_FUNC \
    --model=ssd300_opt2.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_fp32_interpreter.json \
    --count=$1
fi

if [[ $2 -eq 1 ]]; then
$EVAL_FUNC \
    --model=ssd300_quant_int8_per_layer.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_int8_perlayer.json \
    --count=$1  \
    --pre_result_json=./coco_results_int8_perlayer.json

else
$EVAL_FUNC \
    --model=ssd300_quant_int8_per_layer.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_int8_perlayer.json \
    --count=$1
fi

if [[ $2 -eq 1 ]]; then
$EVAL_FUNC \
    --model=ssd300_quant_int8_per_channel.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_int8_per_channel.json \
    --count=$1   \
    --pre_result_json=./coco_results_int8_per_channel.json

else
$EVAL_FUNC \
    --model=ssd300_quant_int8_per_channel.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_int8_per_channel.json \
    --count=$1
fi

if [[ $2 -eq 1 ]]; then
$EVAL_FUNC \
    --model=ssd300_quant_int8_multiplier.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_int8_multiplier.json \
    --count=$1 \
    --pre_result_json=./coco_results_int8_multiplier.json

else
$EVAL_FUNC \
    --model=ssd300_quant_int8_multiplier.mlir \
    --net_input_dims 300,300 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./coco_results_int8_multiplier.json \
    --count=$1
fi