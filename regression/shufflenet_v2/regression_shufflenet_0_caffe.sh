#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel \
    --label_file $REGRESSION_PATH/shufflenet_v2/data/ilsvrc12/synset_words.txt \
    --dump_blobs shufflenet_blobs.npz \
    --dump_weights shufflenet_weights.npz \
    --raw_scale 1.0 \
    $REGRESSION_PATH/shufflenet_v2/data/194.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py shufflenet_blobs.npz shufflenet_in_fp32.npz data
npz_extract.py shufflenet_blobs.npz shufflenet_out_fp32_fc.npz fc

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
#npz_compare.py shufflenet_in_fp32.npz $REGRESSION_PATH/shufflenet_v2/data/shufflenet_in_fp32.npz
cp $REGRESSION_PATH/shufflenet_v2/data/shufflenet_in_fp32.npz shufflenet_in_fp32.npz

# VERDICT
echo $0 PASSED
