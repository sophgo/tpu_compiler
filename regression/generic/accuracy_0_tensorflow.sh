#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "Eval with tensorflow"

if [ "$EVAL_MODEL_TYPE" = "espcn" ]; then
    EVAL_FUNC=eval_tf_espcn.py
    $EVAL_FUNC \
        --model_def $MODEL_DEF \
        --dataset $DATASET_PATH/espcn/ \
        --net_input_dims $NET_INPUT_DIMS \
        --image_resize_dims $IMAGE_RESIZE_DIMS \
        --raw_scale $RAW_SCALE \
        --mean $MEAN \
        --std $STD \
        --input_scale $INPUT_SCALE \
        --model_type tensorflow \
        --model_channel_order $MODEL_CHANNEL_ORDER \
        --count=$1 \
        --gray $BGRAY
fi

echo $0 DONE
