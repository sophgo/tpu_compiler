#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $DIR/data/dog.jpg \
    --mean 0.485,0.456,0.406 \
    --std 0.229,0.224,0.225 \
    --image_resize_dims 256,256 \
    --net_input_dims 224,224 \
    --output_file resnet18_out_onnx.npz \
    --dump_tensor resnet18_out_tensor_all_onnx.npz \
    --model_path $MODEL_PATH/imagenet/resnet/onnx/resnet18.onnx

cvi_npz_tool.py extract resnet18_out_tensor_all_onnx.npz resnet18_in_fp32.npz input
# VERDICT
echo $0 PASSED
