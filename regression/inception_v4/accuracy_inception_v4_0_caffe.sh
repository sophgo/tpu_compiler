#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=eval_caffe_classifier.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/caffe/deploy_inception-v4.prototxt \
    --pretrained_model $MODEL_PATH/caffe/inception-v4.caffemodel \
    --dataset=$DATASET_PATH/imagenet/img_val_extracted \
    --mean 128.0,128.0,128.0 \
    --input_scale 0.0078125 \
    --dim=299 \
    --count=$1

echo $0 DONE
