#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $REGRESSION_PATH/data/pose_256_192.jpg \
    --mean=0.406,0.457,0.48 \
    --net_input_dims 256,192 \
    --image_resize_dims 256,192 \
    --raw_scale=1 \
    --output_file alphapose_res50_out_fp32.npz \
    --model_path $MODEL_PATH/pose/alphapose/onnx/alphapose_resnet50_256x192.onnx \
    --dump_tensor alphapose_res50_out_tensor_all.npz

cvi_npz_tool.py extract alphapose_res50_out_tensor_all.npz alphapose_in_fp32.npz input

# VERDICT
echo $0 PASSED
