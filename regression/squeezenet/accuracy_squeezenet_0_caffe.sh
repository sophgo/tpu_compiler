#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


EVAL_FUNC=eval_classifier.py

$EVAL_FUNC \
    --model_def $MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel \
    --dataset $DATASET_PATH \
    --mean_file $REGRESSION_PATH/data/ilsvrc_2012_mean.npy \
    --count=$1

echo $0 DONE
