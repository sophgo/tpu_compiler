#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CAFFE_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$CAFFE_BLOBS_NPZ" ]; then
  # run caffe model
  run_caffe_classifier.py \
      --model_def $MODEL_DEF \
      --pretrained_model $MODEL_DAT \
      --image_resize_dims $IMAGE_RESIZE_DIMS \
      --net_input_dims $NET_INPUT_DIMS \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --input_scale $INPUT_SCALE \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --batch_size $BATCH_SIZE \
      --label_file $LABEL_FILE \
      --dump_blobs $CAFFE_BLOBS_NPZ \
      $IMAGE_PATH \
      caffe_out.npy
fi

cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_in_fp32.npz $INPUT

if [ ${DO_POSTPROCESS} -eq 1 ]; then
  /bin/bash $POSTPROCESS_SCRIPT $CAFFE_BLOBS_NPZ $OUTPUTS
fi

echo $0 PASSED
