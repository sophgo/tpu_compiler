#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=eval_caffe_classifier.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/caffe/ResNet-50-deploy.prototxt \
    --pretrained_model $MODEL_PATH/caffe/ResNet-50-model.caffemodel \
    --dataset $DATASET_PATH/imagenet/img_val_extracted \
    --mean_file $REGRESSION_PATH/resnet50/data/mean_resize.npy \
    --count=$1

echo $0 DONE
