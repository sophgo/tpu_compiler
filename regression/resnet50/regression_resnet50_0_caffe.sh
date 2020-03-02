#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CAFFE_BLOBS_NPZ="resnet50_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_classifier.py \
      --model_def $MODEL_PATH/imagenet/resnet/caffe/ResNet-50-deploy.prototxt \
      --pretrained_model $MODEL_PATH/imagenet/resnet/caffe/ResNet-50-model.caffemodel \
      --mean_file $REGRESSION_PATH/resnet50/data/ilsvrc_2012_mean.npy \
      --label_file $REGRESSION_PATH/resnet50/data/ilsvrc12/synset_words.txt \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights resnet50_weights.npz \
      $REGRESSION_PATH/resnet50/data/cat.jpg \
      caffe_out.npy
fi

# extract input and output
npz_extract.py $CAFFE_BLOBS_NPZ resnet50_in_fp32.npz input
npz_extract.py $CAFFE_BLOBS_NPZ resnet50_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
npz_compare.py resnet50_in_fp32.npz $REGRESSION_PATH/resnet50/data/resnet50_in_fp32.npz
cp $REGRESSION_PATH/resnet50/data/resnet50_in_fp32.npz resnet50_in_fp32.npz

# VERDICT
echo $0 PASSED
