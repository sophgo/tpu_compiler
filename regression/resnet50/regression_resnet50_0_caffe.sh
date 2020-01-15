#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/caffe/ResNet-50-deploy.prototxt \
    --pretrained_model $MODEL_PATH/caffe/ResNet-50-model.caffemodel \
    --mean_file $PYTHON_TOOLS_PATH/data/ilsvrc_2012_mean.npy \
    --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
    --dump_blobs resnet50_blobs.npz \
    --dump_weights resnet50_weights.npz \
    $REGRESSION_PATH/resnet50/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py resnet50_blobs.npz resnet50_in_fp32.npz input
npz_extract.py resnet50_blobs.npz resnet50_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py resnet50_in_fp32.npz $REGRESSION_PATH/resnet50/data/resnet50_in_fp32.npz
cp $REGRESSION_PATH/resnet50/data/resnet50_in_fp32.npz resnet50_in_fp32.npz

# VERDICT
echo $0 PASSED
