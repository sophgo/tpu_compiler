#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
DATA_DIR=$DIR/data

# run caffe model
# mean = [0.485, 0.456, 0.406] x 255
# std = [0.229, 0.224, 0.225] x 255
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.caffemodel \
    --mean 0.485,0.456,0.406 \
    --raw 1 \
    --input_scale 4.45 \
    --dump_blobs efficientnet_blobs.npz \
    --dump_weights efficientnet_weights.npz \
    $REGRESSION_PATH/resnet50/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_tool.py extract efficientnet_blobs.npz efficientnet_in_fp32.npz data
npz_tool.py extract efficientnet_blobs.npz efficientnet_out_fp32_prob.npz _fc

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_tool.py compare efficientnet_in_fp32.npz $REGRESSION_PATH/efficientnet_b0/data/efficientnet_in_fp32.npz
cp $REGRESSION_PATH/efficientnet_b0/data/efficientnet_in_fp32.npz efficientnet_in_fp32.npz

# VERDICT
echo $0 PASSED
