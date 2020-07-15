#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_detector_yolo_v1.py \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --net_input_dims $NET_INPUT_DIMS \
      --obj_threshold 0.15 \
      --nms_threshold 0.5 \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      --batch_size $BATCH_SIZE \
      --input_file $REGRESSION_PATH/data/dog.jpg \
      --draw_image dog_out.jpg
fi

# extract input
cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz data

# VERDICT
echo $0 PASSED
