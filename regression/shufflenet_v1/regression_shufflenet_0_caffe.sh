#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3.caffemodel \
    --label_file $REGRESSION_PATH/data/synset_words.txt \
    --mean 103.94,116.78,123.68 \
    --input_scale 0.017 \
    --dump_blobs shufflenet_blobs.npz \
    --dump_weights shufflenet_weights.npz \
    $REGRESSION_PATH/data/cat.jpg \
    caffe_out.npy

# extract input and output
cvi_npz_tool.py extract shufflenet_blobs.npz shufflenet_in_fp32.npz input
cvi_npz_tool.py extract shufflenet_blobs.npz shufflenet_out_fp32_prob.npz fc1000

# VERDICT
echo $0 PASSED
