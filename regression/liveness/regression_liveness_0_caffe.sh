#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CAFFE_BLOBS_NPZ="liveness_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_feature_extract.py \
      --model_def $MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.prototxt \
      --pretrained_model $MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.caffemodel \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights liveness_weights.npz \
      --model_type liveness \
      --input_file $REGRESSION_PATH/liveness/data/liveness_1_patch.bin
fi

# extract input and output
npz_tool.py extract $CAFFE_BLOBS_NPZ liveness_in_fp32.npz data
npz_tool.py extract $CAFFE_BLOBS_NPZ liveness_out_fp32_prob.npz fc2

# VERDICT
echo $0 PASSED
