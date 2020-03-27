#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# fuse-relu
mlir-opt \
    --fuse-relu \
    squeezenet_opt.mlir \
    -o squeezenet_opt_fuse_relu.mlir

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    squeezenet_opt_fuse_relu.mlir \
    -o squeezenet_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter squeezenet_quant_bf16.mlir \
    --tensor-in squeezenet_in_fp32.npz \
    --tensor-out squeezenet_out_bf16.npz \
    --dump-all-tensor=squeezenet_tensor_all_bf16.npz
cvi_npz_tool.py compare squeezenet_out_bf16.npz squeezenet_out_fp32.npz -v
cvi_npz_tool.py compare \
    squeezenet_tensor_all_bf16.npz \
    squeezenet_tensor_all_fp32.npz \
    --op_info squeezenet_op_info.csv \
    --tolerance=0.99,0.99,0.93 -vv

# VERDICT
echo $0 PASSED
