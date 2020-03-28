#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/densenet/caffe/densenet121_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/densenet/caffe/densenet121.caffemodel \
    --mean 103.94,116.78,123.68 \
    --input_scale 0.017 \
    --net_input_dims 224,224 \
    --dump_blobs densenet_blobs.npz \
    --dump_weights densenet_weights.npz \
    --label_file $REGRESSION_PATH/data/synset_words.txt \
    $REGRESSION_PATH/data/cat.jpg \
    caffe_out.npy


    # --force_input calibration_input.npy \

# extract input and output
cvi_npz_tool.py extract densenet_blobs.npz densenet_in_fp32.npz input
cvi_npz_tool.py extract densenet_blobs.npz densenet_out_fp32_fc6.npz fc6

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
#cvi_npz_tool.py compare densenet121_in_fp32.npz $REGRESSION_PATH/densenet121/data/densenet121_in_fp32.npz
#cp $REGRESSION_PATH/densenet121/data/densenet121_in_fp32.npz densenet121_in_fp32.npz

# VERDICT
echo $0 PASSED