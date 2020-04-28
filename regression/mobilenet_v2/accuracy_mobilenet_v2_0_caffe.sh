#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


EVAL_FUNC=eval_classifier.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel \
    --dataset $DATASET_PATH/imagenet/img_val_extracted \
    --mean 103.94,116.78,123.68 \
    --input_scale 0.017 \
    --count=$1

echo $0 DONE
