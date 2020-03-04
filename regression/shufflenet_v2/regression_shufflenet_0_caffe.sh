#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

#genarate input
#python ./python/convert_image.py --image data/194.jpg --save data/shufflenet_in_fp32

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/shufflenet_v2/caffe/shufflenet_v2_x0.5.caffemodel \
    --label_file $REGRESSION_PATH/shufflenet_v2/data/ilsvrc12/synset_words.txt \
    --dump_blobs shufflenet_blobs.npz \
    --dump_weights shufflenet_weights.npz \
    --force_input $REGRESSION_PATH/shufflenet_v2/data/shufflenet_in_fp32.npy \
    $REGRESSION_PATH/shufflenet_v2/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py shufflenet_blobs.npz shufflenet_in_fp32.npz data
npz_extract.py shufflenet_blobs.npz shufflenet_out_fp32_fc.npz fc

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py shufflenet_in_fp32.npz $REGRESSION_PATH/shufflenet_v2/data/shufflenet_in_fp32.npz
cp $REGRESSION_PATH/shufflenet_v2/data/shufflenet_in_fp32.npz shufflenet_in_fp32.npz

# VERDICT
echo $0 PASSED
