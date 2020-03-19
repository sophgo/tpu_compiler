#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_bf16_per_layer.csv \
    ${NET}_opt.mlir \
    -o ${NET}_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter ${NET}_quant_bf16.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_bf16.npz \
    --dump-all-tensor=${NET}_tensor_all_bf16.npz

npz_tool.py compare \
    ${NET}_tensor_all_bf16.npz \
    ${NET}_tensor_all_fp32.npz \
    --op_info ${NET}_op_info.csv \
    --tolerance=0.99,0.99,0.94 -vv

# VERDICT
echo $0 PASSED
