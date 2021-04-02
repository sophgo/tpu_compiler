#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


ONNX_BLOBS_NPZ="${NET}_blobs.npz"

if [ ! -f "$ONNX_BLOBS_NPZ" ]; then
  run_onnx_rnn_toy.py \
      --model_path $MODEL_DEF \
      --batch_size $BATCH_SIZE \
      --dump_tensor $ONNX_BLOBS_NPZ \
      --input_file $REGRESSION_PATH/data/ocr_in_fp32.npz \
      --output_file onnx_out.npz
fi

cvi_npz_tool.py extract $ONNX_BLOBS_NPZ ${NET}_in_fp32.npz input
# cvi_npz_tool.py extract $CAFFE_BLOBS_NPZ ${NET}_out_fp32_prob.npz output

# VERDICT
echo $0 PASSED
