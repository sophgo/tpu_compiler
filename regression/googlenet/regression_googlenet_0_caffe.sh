#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/googlenet/caffe/deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel \
    --label_file $REGRESSION_PATH/data/synset_words.txt \
    --mean 104,117,123 \
    --dump_blobs googlenet_blobs.npz \
    --dump_weights googlenet_weights.npz \
    $REGRESSION_PATH/data/cat.jpg \
    caffe_out.npy

# extract input and output
cvi_npz_tool.py extract googlenet_blobs.npz googlenet_in_fp32.npz data
cvi_npz_tool.py extract googlenet_blobs.npz googlenet_out_fp32_prob.npz prob

# VERDICT
echo $0 PASSED
