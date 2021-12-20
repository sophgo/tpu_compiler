#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "Eval with onnxruntime"

if [ "$EVAL_MODEL_TYPE" = "isbi" ]; then
    EVAL_FUNC=eval_unet.py
    $EVAL_FUNC \
        --model_def $MODEL_DEF \
        --dataset $DATASET_PATH/unet/ \
        --net_input_dims $NET_INPUT_DIMS \
        --image_resize_dims $IMAGE_RESIZE_DIMS \
        --raw_scale $RAW_SCALE \
        --mean $MEAN \
        --std $STD \
        --input_scale $INPUT_SCALE \
        --model_type onnx \
        --model_channel_order $MODEL_CHANNEL_ORDER \
        --count=$1
elif [ "$EVAL_MODEL_TYPE" = "coco" ]; then
    EVAL_FUNC=$EVAL_SCRIPT_ONNX
    # val onnx
    $EVAL_FUNC \
    --model $MODEL_DEF \
    --net_input_dims ${NET_INPUT_DIMS} \
    --coco_image_path $DATASET_PATH/coco/val2017 \
    --coco_annotation=$DATASET_PATH/coco/annotations/instances_val2017.json \
    --coco_result_jason_file restult_${NET}_onnx.json \
    --count=$1 \

else
    EVAL_FUNC=eval_classifier.py

    $EVAL_FUNC \
        --model_def $MODEL_DEF \
        --dataset $DATASET_PATH/imagenet/img_val_extracted \
        --net_input_dims $NET_INPUT_DIMS \
        --image_resize_dims $IMAGE_RESIZE_DIMS \
        --raw_scale $RAW_SCALE \
        --mean $MEAN \
        --std $STD \
        --input_scale $INPUT_SCALE \
        --model_type onnx \
        --model_channel_order $MODEL_CHANNEL_ORDER \
        --count=$1
fi

echo $0 DONE
