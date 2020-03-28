#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CAFFE_BLOBS_NPZ="bmface_v3_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_feature_extract.py \
      --model_def $MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.prototxt \
      --pretrained_model $MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.caffemodel \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights bmface_v3_weights.npz \
      --model_type bmface_v3 \
      --input_file $REGRESSION_PATH/data/Aaron_Eckhart_0001.jpg
fi

# extract input and output
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ bmface_v3_in_fp32.npz data
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ bmface_v3_out_fp32.npz fc1

# VERDICT
echo $0 PASSED
