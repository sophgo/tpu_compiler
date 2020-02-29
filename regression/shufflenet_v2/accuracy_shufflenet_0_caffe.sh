#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

EVAL_FUNC=eval_caffe_classifier.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5_fixed.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel \
    --dataset $DATASET_PATH/imagenet/img_val_extracted \
    --mean_file $REGRESSION_PATH/shufflenet_v2/data/mean_resize.npy \
    --count=$1

echo $0 DONE
