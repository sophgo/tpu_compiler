#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"



# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $DIR/data/cat.npz \
    --output_file efficientnet_b0_out_onnx.npz \
    --model_path $MODEL_PATH/imagenet/efficientnet-b0/onnx/efficientnet_b0.onnx


cvi_npz_tool.py extract $DIR/data/cat.npz efficientnet_b0_in_fp32.npz input
# VERDICT
echo $0 PASSED
