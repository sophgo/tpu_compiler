#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    vgg16_opt.mlir \
    -o vgg16_opt2.mlir

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    vgg16_opt2.mlir \
    -o vgg16_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter vgg16_quant_bf16.mlir \
    --tensor-in vgg16_in_fp32.npz \
    --tensor-out vgg16_out_bf16.npz \
    --dump-all-tensor=vgg16_tensor_all_bf16.npz
npz_compare.py vgg16_out_bf16.npz vgg16_out_fp32.npz -v
npz_compare.py \
    vgg16_tensor_all_bf16.npz \
    vgg16_tensor_all_fp32.npz \
    --op_info vgg16_op_info.csv \
    --tolerance=0.99,0.99,0.97 -vvv

# VERDICT
echo $0 PASSED
