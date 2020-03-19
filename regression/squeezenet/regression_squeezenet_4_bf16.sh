#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# fuse-relu
mlir-opt \
    --fuse-relu \
    squeezenet_v1.1_opt.mlir \
    -o squeezenet_v1.1_opt_fuse_relu.mlir

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    squeezenet_v1.1_opt_fuse_relu.mlir \
    -o squeezenet_v1.1_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter squeezenet_v1.1_quant_bf16.mlir \
    --tensor-in squeezenet_v1.1_in_fp32.npz \
    --tensor-out squeezenet_v1.1_out_bf16.npz \
    --dump-all-tensor=squeezenet_v1.1_tensor_all_bf16.npz
cvi_npz_tool.py compare squeezenet_v1.1_out_bf16.npz squeezenet_v1.1_out_fp32.npz -v
cvi_npz_tool.py compare \
    squeezenet_v1.1_tensor_all_bf16.npz \
    squeezenet_v1.1_tensor_all_fp32.npz \
    --op_info squeezenet_v1.1_op_info.csv \
    --tolerance=0.99,0.99,0.93 -vv

# VERDICT
echo $0 PASSED
