#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    shufflenet_opt.mlir \
    -o shufflenet_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter shufflenet_quant_bf16.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_bf16.npz \
    --dump-all-tensor=shufflenet_tensor_all_bf16.npz
npz_compare.py shufflenet_out_bf16.npz shufflenet_out_fp32.npz -v
npz_compare.py \
    shufflenet_tensor_all_bf16.npz \
    shufflenet_tensor_all_fp32.npz \
    --op_info shufflenet_op_info.csv \
    --tolerance=0.99,0.99,0.93 -vv

# VERDICT
echo $0 PASSED
