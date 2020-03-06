#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CAFFE_BLOBS_NPZ="bmface-v3_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_feature_extract.py \
      --model_def $MODEL_PATH/face_recognition/bmface/caffe/fp32/2020.01.15.01/bmface-v3.prototxt \
      --pretrained_model $MODEL_PATH/face_recognition/bmface/caffe/fp32/2020.01.15.01/bmface-v3.caffemodel \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --dump_weights bmface-v3_weights.npz \
      --input_file $REGRESSION_PATH/bmface_v3/data/Aaron_Eckhart_0001.jpg 
fi

# extract input and output
npz_extract.py $CAFFE_BLOBS_NPZ bmface-v3_in_fp32.npz data
npz_extract.py $CAFFE_BLOBS_NPZ bmface-v3_out_fp32_prob.npz fc1

# VERDICT
echo $0 PASSED
