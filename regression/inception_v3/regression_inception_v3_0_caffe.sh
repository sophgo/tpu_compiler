#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


echo $0 is RUNNING

CAFFE_BLOBS_NPZ="inception_v3_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_classifier.py \
      --model_def $MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt \
      --pretrained_model $MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel \
      --net_input_dims 299,299 \
      --mean 128,128,128 \
      --input_scale 0.0078125 \
      --label_file $REGRESSION_PATH/data/ilsvrc2015_synset_words.txt \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights inception_v3_weights.npz \
      $REGRESSION_PATH/data/cat.jpg \
      caffe_out.npy
fi

# extract input and output
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ inception_v3_in_raw_fp32.npz raw_data
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ inception_v3_out_fp32_prob.npz prob

# VERDICT
echo $0 PASSED
