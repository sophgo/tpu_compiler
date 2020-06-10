#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

run_onnx_espcn.py \
    --model_def $MODEL_DEF \
    --input_file $REGRESSION_PATH/data/BSD100_001.png \
    --output_file ${NET}_out_fp32_ref.npz \
    --net_input_dims $NET_INPUT_DIMS \
    --dump_tensor ${NET}_blobs.npz \
    --batch_size $BATCH_SIZE

# extract input and output
cvi_npz_tool.py extract ${NET}_blobs.npz ${NET}_in_fp32.npz input

# VERDICT
echo $0 PASSED