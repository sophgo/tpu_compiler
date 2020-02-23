#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel \
    --mean 103.94,116.78,123.68 \
    --input_scale 0.017 \
    --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
    --dump_blobs mobilenet_v2_blobs.npz \
    --dump_weights mobilenet_v2_weights.npz \
    $REGRESSION_PATH/resnet50/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py mobilenet_v2_blobs.npz mobilenet_v2_in_fp32.npz input
npz_extract.py mobilenet_v2_blobs.npz mobilenet_v2_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py mobilenet_v2_in_fp32.npz $REGRESSION_PATH/mobilenet_v2/data/mobilenet_v2_in_fp32.npz
cp $REGRESSION_PATH/mobilenet_v2/data/mobilenet_v2_in_fp32.npz mobilenet_v2_in_fp32.npz

# VERDICT
echo $0 PASSED
