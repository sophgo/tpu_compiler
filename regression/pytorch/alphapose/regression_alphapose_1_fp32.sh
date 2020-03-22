#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# use python to mlir , gen golden too
cvi_model_convert.py \
    --model_path $MODEL_PATH/pose/alphapose/onnx/alphapose_resnet50_256x192.onnx \
    --model_name alphapose \
    --model_type onnx \
    --mlir_file_path alphapose.mlir


mlir-opt \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --print-tpu-op-info \
    alphapose.mlir \
    -o alphapose_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter alphapose_opt.mlir \
    --tensor-in alphapose_in_fp32.npz \
    --tensor-out alphapose_out_fp32.npz \
    --dump-all-tensor=alphapose_tensor_all_fp32.npz
cvi_npz_tool.py rename alphapose_out_fp32.npz output_Conv output
cvi_npz_tool.py compare alphapose_out_fp32.npz alphapose_res50_out_fp32.npz \
    --tolerance=0.9,0.9,-0.9 -vv


# VERDICT
echo $0 PASSED
