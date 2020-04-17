#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"



# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $REGRESSION_PATH/data/dog.jpg \
    --mean 0.485,0.456,0.406 \
    --image_resize_dims 224,224 \
    --net_input_dims 224,224 \
    --std 0.229,0.224,0.225 \
    --raw_scale 1 \
    --output_file efficientnet_b0_out_onnx.npz \
    --dump_tensor efficientnet_b0_out_tensor_all_onnx.npz \
    --model_path $MODEL_PATH/imagenet/efficientnet-b0/onnx/efficientnet_b0.onnx

cvi_npz_tool.py extract efficientnet_b0_out_tensor_all_onnx.npz efficientnet_b0_in_fp32.npz input
# VERDICT
echo $0 PASSED
