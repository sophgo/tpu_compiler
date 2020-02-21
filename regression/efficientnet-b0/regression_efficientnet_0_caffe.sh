#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
DATA_DIR=$DIR/data

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/caffe/efficientnet-b0.prototxt \
    --pretrained_model $MODEL_PATH/caffe/efficientnet-b0.caffemodel \
    --mean 0.485,0.456,0.406 \
    --raw 1 \
    $REGRESSION_PATH/efficientnet-b0/data/husky.jpg  \
    caffe_out.npy

# extract input and output
npz_extract.py efficientnet_blobs.npz efficientnet_in_fp32.npz data
npz_extract.py efficientnet_blobs.npz efficientnet_out_fp32_prob.npz _fc

# VERDICT
echo $0 PASSED
