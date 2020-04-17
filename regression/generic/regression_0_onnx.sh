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
      --input_scale ${INPUT_SCALE} \
      --dump_tensor $ONNX_BLOBS_NPZ \
      --input_file $REGRESSION_PATH/data/cat.jpg \
      --output_file onnx_out.npz
fi

cvi_npz_tool.py extract $ONNX_BLOBS_NPZ ${NET}_in_fp32.npz input
# cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32_prob.npz output

# VERDICT
echo $0 PASSED
