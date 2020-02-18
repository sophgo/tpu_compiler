#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# quantization
mlir-opt \
    --quant-bf16 \
    resnet50_opt.mlir \
    -o resnet50_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter resnet50_quant_bf16.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_bf16.npz \
    --dump-all-tensor=resnet50_tensor_all_bf16.npz
npz_compare.py resnet50_out_bf16.npz resnet50_out_fp32.npz -v
npz_compare.py \
    resnet50_tensor_all_bf16.npz \
    resnet50_tensor_all_fp32.npz \
    --op_info resnet50_op_info.csv \
    --tolerance=0.99,0.99,0.95 -vvv

# VERDICT
echo $0 PASSED
