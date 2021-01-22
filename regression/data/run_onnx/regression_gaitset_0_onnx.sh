#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $IMAGE_PATH \
    --net_input_dims ${NET_INPUT_DIMS} \
    --batch ${BATCH_SIZE} \
    --raw_scale ${RAW_SCALE} \
    --mean ${MEAN} \
    --input_scale ${INPUT_SCALE} \
    --std ${STD} \
    --output_file ${NET}_out_fp32_ref.npz \
    --model_path $MODEL_DEF \
    --dump_tensor ${NET}_blobs.npz \
    --gray $BGRAY

cvi_npz_tool.py extract ${NET}_blobs.npz ${NET}_in_fp32.npz input

# VERDICT
echo $0 PASSED