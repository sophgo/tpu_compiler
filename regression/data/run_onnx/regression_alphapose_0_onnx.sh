#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $REGRESSION_PATH/data/alphapose_in.npz \
    --output_file ${NET}_out_fp32_ref.npz \
    --model_path $MODEL_DEF

cvi_npz_tool.py extract $REGRESSION_PATH/data/alphapose_in.npz ${NET}_in_fp32.npz input
cp ${NET}_out_fp32_ref.npz ${NET}_blobs.npz

# VERDICT
echo $0 PASSED
