#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1
MODEL=$2


# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $DIR/data/cat.npz \
    --output_file ${NET}_out_onnx.npz \
    --model_path $MODEL


cvi_npz_tool.py extract $DIR/data/cat.npz ${NET}_in_fp32.npz input
# VERDICT
echo $0 PASSED
