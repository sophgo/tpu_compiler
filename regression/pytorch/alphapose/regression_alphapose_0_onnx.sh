#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../../envsetup.sh

# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $REGRESSION_PATH/pytorch/alphapose/data/pose.npz \
    --output_file alphapose_res50_out_fp32.npz \
    --model_path $MODEL_PATH/pose/alphapose/onnx/alphapose_resnet50_256x192.onnx

npz_extract.py $REGRESSION_PATH/pytorch/alphapose/data/pose.npz alphapose_in_fp32.npz input

# VERDICT
echo $0 PASSED
