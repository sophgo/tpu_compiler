#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/squeezenet/caffe/squeezenet_v1.1.caffemodel \
    --mean_file $PYTHON_TOOLS_PATH/data/ilsvrc_2012_mean.npy \
    --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
    --dump_blobs squeezenet_v1.1_blobs.npz \
    --dump_weights squeezenet_v1.1_weights.npz \
    $REGRESSION_PATH/squeezenet/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py squeezenet_v1.1_blobs.npz squeezenet_v1.1_in_fp32.npz input
npz_extract.py squeezenet_v1.1_blobs.npz squeezenet_v1.1_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py squeezenet_v1.1_in_fp32.npz $REGRESSION_PATH/squeezenet/data/squeezenet_v1.1_in_fp32.npz
cp $REGRESSION_PATH/squeezenet/data/squeezenet_v1.1_in_fp32.npz squeezenet_v1.1_in_fp32.npz

# VERDICT
echo $0 PASSED
