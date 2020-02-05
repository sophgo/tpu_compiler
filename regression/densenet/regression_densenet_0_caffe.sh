#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/densenet/caffe/densenet121_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/densenet/caffe/densenet121.caffemodel \
    --mean 103.94,116.78,123.68 \
    --input_scale 0.017 \
    --dump_blobs densenet_blobs.npz \
    --dump_weights densenet_weights.npz \
    --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
    $REGRESSION_PATH/densenet/data/cat.jpg \
    caffe_out.npy


# extract input and output
npz_extract.py densenet_blobs.npz densenet_in_fp32.npz input
npz_extract.py densenet_blobs.npz densenet_out_fp32_prob.npz fc6

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
#npz_compare.py densenet121_in_fp32.npz $REGRESSION_PATH/densenet121/data/densenet121_in_fp32.npz
#cp $REGRESSION_PATH/densenet121/data/densenet121_in_fp32.npz densenet121_in_fp32.npz

# VERDICT
echo $0 PASSED