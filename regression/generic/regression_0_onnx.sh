#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


ONNX_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$ONNX_BLOBS_NPZ" ]; then
  run_onnx_inference.py \
      --model_path $MODEL_DEF \
      --image_resize_dims ${IMAGE_RESIZE_DIMS} \
      --net_input_dims ${NET_INPUT_DIMS} \
      --raw_scale ${RAW_SCALE} \
      --mean ${MEAN} \
      --std ${STD} \
      --batch_size $BATCH_SIZE \
      --input_scale ${INPUT_SCALE} \
      --dump_tensor $ONNX_BLOBS_NPZ \
      --input_file $IMAGE_PATH \
      --model_channel_order $MODEL_CHANNEL_ORDER \
      --output_file onnx_out.npz
fi

cvi_npz_tool.py extract $ONNX_BLOBS_NPZ ${NET}_in_fp32.npz input
echo $0 PASSED
