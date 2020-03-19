#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    densenet_opt.mlir \
    -o densenet_opt2.mlir

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    densenet_opt2.mlir \
    -o densenet_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter densenet_quant_bf16.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_bf16.npz \
    --dump-all-tensor=densenet_tensor_all_bf16.npz
npz_tool.py compare densenet_out_bf16.npz densenet_out_fp32.npz -v
npz_tool.py compare \
    densenet_tensor_all_bf16.npz \
    densenet_tensor_all_fp32.npz \
    --op_info densenet_op_info.csv \
    --tolerance=0.99,0.99,0.94 -vvv

# VERDICT
echo $0 PASSED
