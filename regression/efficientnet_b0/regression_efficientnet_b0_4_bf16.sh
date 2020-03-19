#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    efficientnet_b0_opt.mlir \
    -o efficientnet_b0_quant_bf16.mlir

# create bf16 input
cvi_npz_tool.py to_bin $REGRESSION_PATH/efficientnet_b0/data/efficientnet_in_fp32.npz data efficientnet_in_fp32.bin
bin_fp32_to_bf16.py \
    efficientnet_in_fp32.bin \
    efficientnet_in_bf16.bin \
    1.0

# bf16 inference
mlir-tpu-interpreter efficientnet_b0_quant_bf16.mlir\
    --tensor-in $REGRESSION_PATH/efficientnet_b0/data/efficientnet_in_fp32.npz \
    --tensor-out efficientnet_out_bf16.npz \
    --dump-all-tensor=efficientnet_tensor_all_bf16.npz

cvi_npz_tool.py compare ./efficientnet_tensor_all_bf16.npz ./efficientnet_tensor_all_fp32.npz -v

# need to check torlerance later
cvi_npz_tool.py compare \
    ./efficientnet_tensor_all_bf16.npz\
    ./efficientnet_tensor_all_fp32.npz \
    --tolerance=0.99,0.99,0.88 -vvv

# VERDICT
echo $0 PASSED
