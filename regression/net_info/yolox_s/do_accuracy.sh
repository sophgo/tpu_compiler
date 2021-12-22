#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

count=0
if [ x$1 != x ]; then
    count=$1
fi
source
EVAL_FUNC=${MLIR_SRC_PATH}/python/cvi_toolkit/eval/eval_yolox.py
eval_yolox
# caffe eval
echo "eval with onnx"
$EVAL_FUNC \
    --model $MODEL_PATH/object_detection/yolox/onnx/yolox_s.onnx \
    --net_input_dims 640,640 \
    --coco_image_path $DATASET_PATH/coco/val2017 \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file restult_yolox_s_onnx.json \
    --count=$1 \

echo  "eval with mlir"

$_EVAL_FUNC \
    --model=yolox_s_quant_int8_multiplier.mlir \
    --net_input_dims 640,640 \
    --coco_image_path=$DATASET_PATH/coco/val2017/ \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file=./yolox_s_coco_results_int8_multiplier.json \
    --count=$1

# or just run acc
# accuracy_generic.sh yolox_s $1
echo $0 DONE



