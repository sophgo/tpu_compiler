#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/squeezenet/caffe/deploy_v1.1.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel \
    --mean_file $REGRESSION_PATH/squeezenet/data/ilsvrc_2012_mean.npy \
    --label_file $REGRESSION_PATH/data/synset_words.txt \
    --dump_blobs squeezenet_blobs.npz \
    --dump_weights squeezenet_weights.npz \
    $REGRESSION_PATH/data/cat.jpg \
    caffe_out.npy

# extract input and output
cvi_npz_tool.py extract squeezenet_blobs.npz squeezenet_in_fp32.npz data
cvi_npz_tool.py extract squeezenet_blobs.npz squeezenet_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
#cvi_npz_tool.py compare squeezenet_in_fp32.npz $REGRESSION_PATH/squeezenet/data/squeezenet_in_fp32.npz
#cp $REGRESSION_PATH/squeezenet/data/squeezenet_in_fp32.npz squeezenet_in_fp32.npz

# VERDICT
echo $0 PASSED
