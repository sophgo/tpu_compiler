#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


EVAL_FUNC=eval_classifier.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel \
    --dataset $DATASET_PATH/imagenet/img_val_extracted \
    --mean 103.94,116.78,123.68 \
    --count=$1

echo $0 DONE
