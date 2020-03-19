#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../../envsetup.sh


convert.py \
    --model_path $MODEL_PATH/imagenet/efficientnet-b0/onnx/efficientnet_b0.onnx \
    --model_name efficientnet_b0 \
    --model_type onnx \
    --mlir_file_path efficientnet_b0.mlir

mlir-opt \
    --assign-layer-id \
    --canonicalize \
    --convert-bn-to-scale \
    --print-tpu-op-info \
    --tpu-op-info-filename efficientnet_b0_op_info.csv \
    efficientnet_b0.mlir \
    -o efficientnet_b0_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter efficientnet_b0_opt.mlir \
    --tensor-in efficientnet_b0_in_fp32.npz \
    --tensor-out efficientnet_b0_out_fp32.npz \
    --dump-all-tensor=efficientnet_b0_tensor_all_fp32.npz

# rename onnx output
npz_tool.py rename \
    efficientnet_b0_out_fp32.npz \
    output_Gemm \
    output

npz_tool.py compare \
    efficientnet_b0_out_fp32.npz \
    efficientnet_b0_out_onnx.npz -vvv


# VERDICT
echo $0 PASSED
