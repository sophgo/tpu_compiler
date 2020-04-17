#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1
MODEL=$2


# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $REGRESSION_PATH/data/cat.jpg \
    --mean 0.485,0.456,0.406 \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --std 0.229,0.224,0.225 \
    --raw_scale 1 \
    --output_file ${NET}_out_onnx.npz \
    --dump_tensor ${NET}_out_tensor_all_onnx.npz \
    --model_path $MODEL


cvi_npz_tool.py extract ${NET}_out_tensor_all_onnx.npz ${NET}_in_fp32.npz input
# VERDICT
echo $0 PASSED
