#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

echo $0 is RUNNING
# run caffe model
run_caffe_classifier.py \
    --model_def $MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt \
    --pretrained_model $MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel \
    --images_dim 299,299 \
    --mean 128,128,128 \
    --input_scale 0.0078125 \
    --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
    --dump_blobs inception_v4_blobs.npz \
    --dump_weights inception_v4_weights.npz \
    $REGRESSION_PATH/inception_v4/data/cat.jpg \
    caffe_out.npy

# extract input and output
npz_extract.py inception_v4_blobs.npz inception_v4_in_fp32.npz input
npz_extract.py inception_v4_blobs.npz inception_v4_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py inception_v4_in_fp32.npz $REGRESSION_PATH/inception_v4/data/inception_v4_in_fp32.npz
cp $REGRESSION_PATH/inception_v4/data/inception_v4_in_fp32.npz inception_v4_in_fp32.npz

# VERDICT
echo $0 PASSED
