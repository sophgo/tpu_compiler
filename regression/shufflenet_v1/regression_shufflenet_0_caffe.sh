#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3.caffemodel \
    --mean_file $PYTHON_TOOLS_PATH/data/ilsvrc_2012_mean.npy \
    --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
    --dump_blobs shufflenet_blobs.npz \
    --dump_weights shufflenet_weights.npz \
    $REGRESSION_PATH/shufflenet_v1/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py shufflenet_blobs.npz shufflenet_in_fp32.npz input
npz_extract.py shufflenet_blobs.npz shufflenet_out_fp32_prob.npz fc1000

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py shufflenet_in_fp32.npz $REGRESSION_PATH/shufflenet_v1/data/shufflenet_in_fp32.npz
cp $REGRESSION_PATH/shufflenet_v1/data/shufflenet_in_fp32.npz shufflenet_in_fp32.npz

# VERDICT
echo $0 PASSED
