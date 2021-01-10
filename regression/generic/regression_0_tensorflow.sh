#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


TF_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$TF_BLOBS_NPZ" ]; then
  cvi_model_inference.py \
      --model_def $MODEL_DEF \
      --image_resize_dims ${IMAGE_RESIZE_DIMS} \
      --net_input_dims ${NET_INPUT_DIMS} \
      --raw_scale ${RAW_SCALE} \
      --mean ${MEAN} \
      --std ${STD} \
      --batch_size $BATCH_SIZE \
      --input_scale ${INPUT_SCALE} \
      --dump_tensor $TF_BLOBS_NPZ \
      --input_file $IMAGE_PATH \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --model_type tensorflow \
      --output_file tf_out.npz \
      --gray $BGRAY

      cvi_npz_tool.py tranpose $TF_BLOBS_NPZ nhwc nchw
      cvi_npz_tool.py extract $TF_BLOBS_NPZ ${NET}_in_fp32.npz input
fi


# VERDICT
echo $0 PASSED
