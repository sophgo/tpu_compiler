#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../../envsetup.sh

# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $DIR/data/dog.npz \
    --output_file resnet18_out_onnx.npz \
    --model_path $MODEL_PATH/imagenet/resnet/onnx/resnet18.onnx

cvi_npz_tool.py extract $DIR/data/dog.npz resnet18_in_fp32.npz input
# VERDICT
echo $0 PASSED
