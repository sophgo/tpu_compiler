#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


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

echo $0 DONE
