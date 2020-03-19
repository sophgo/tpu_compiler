#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel \
    --mean 103.94,116.78,123.68 \
    --dump_blobs vgg16_blobs.npz \
    --dump_weights vgg16_weights.npz \
    $REGRESSION_PATH/resnet50/data/cat.jpg \
    caffe_out.npy

# extract input and output
cvi_npz_tool.py extract vgg16_blobs.npz vgg16_in_fp32.npz input
cvi_npz_tool.py extract vgg16_blobs.npz vgg16_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
cvi_npz_tool.py compare vgg16_in_fp32.npz $REGRESSION_PATH/vgg16/data/vgg16_in_fp32.npz
cp $REGRESSION_PATH/vgg16/data/vgg16_in_fp32.npz vgg16_in_fp32.npz

# VERDICT
echo $0 PASSED
