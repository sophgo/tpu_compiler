#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


EVAL_FUNC=eval_classifier.py

$EVAL_FUNC \
    --model_def $MODEL_DEF \
    --pretrained_model $MODEL_DAT \
    --dataset $DATASET_PATH/imagenet/img_val_extracted \
    --net_input_dims $NET_INPUT_DIMS \
    --image_resize_dims $IMAGE_RESIZE_DIMS \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --input_scale $INPUT_SCALE \
    --count=$1

echo $0 DONE
