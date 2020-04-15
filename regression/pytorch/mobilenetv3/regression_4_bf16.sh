#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1


# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    ${NET}_opt.mlir \
    -o ${NET}_opt2.mlir

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    ${NET}_opt2.mlir \
    -o ${NET}_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter ${NET}_quant_bf16.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_bf16.npz \
    --dump-all-tensor=${NET}_tensor_all_bf16.npz
cvi_npz_tool.py compare ${NET}_out_bf16.npz ${NET}_out_fp32.npz -v
cvi_npz_tool.py compare \
    ${NET}_tensor_all_bf16.npz \
    ${NET}_tensor_all_fp32.npz \
    --op_info ${NET}_op_info.csv \
    --tolerance=0.99,0.99,0.93 -vvv

# VERDICT
echo $0 PASSED
