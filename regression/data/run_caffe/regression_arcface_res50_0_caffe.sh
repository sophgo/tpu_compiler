#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CAFFE_BLOBS_NPZ="arcface_res50_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_feature_extract.py \
      --model_def $MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.prototxt \
      --pretrained_model $MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.caffemodel \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights arcface_res50_weights.npz \
      --dump_blobs_with_inplace=1 \
      --model_type arcface_res50 \
      --batch_size $BATCH_SIZE \
      --input_file $REGRESSION_PATH/data/Aaron_Eckhart_0001.jpg
fi

# extract input and output
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ arcface_res50_in_fp32.npz data
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ arcface_res50_out_fp32_prob.npz fc1

# VERDICT
echo $0 PASSED
