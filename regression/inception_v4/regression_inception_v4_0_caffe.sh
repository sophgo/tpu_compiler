#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


echo $0 is RUNNING

CAFFE_BLOBS_NPZ="inception_v4_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_classifier.py \
      --model_def $MODEL_PATH/imagenet/inception_v4/caffe/deploy_inception-v4.prototxt \
      --pretrained_model $MODEL_PATH/imagenet/inception_v4/caffe/inception-v4.caffemodel \
      --net_input_dims 299,299 \
      --mean 128,128,128 \
      --input_scale 0.0078125 \
      --label_file $PYTHON_TOOLS_PATH/data/ilsvrc12/synset_words.txt \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights inception_v4_weights.npz \
      $REGRESSION_PATH/inception_v4/data/cat.jpg \
      caffe_out.npy
fi

# extract input and output
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ inception_v4_in_fp32.npz input
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ inception_v4_out_fp32_prob.npz prob

# fix input data consistency
# because jpeg decoder may introduce difference, use save file to overwrite
cvi_npz_tool.py compare inception_v4_in_fp32.npz $REGRESSION_PATH/inception_v4/data/inception_v4_in_fp32.npz
cp $REGRESSION_PATH/inception_v4/data/inception_v4_in_fp32.npz inception_v4_in_fp32.npz

# VERDICT
echo $0 PASSED
