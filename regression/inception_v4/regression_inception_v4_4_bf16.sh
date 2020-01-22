#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    inception_v4_opt.mlir \
    -o inception_v4_opt2.mlir

# quantization
mlir-opt \
    --quant-bf16 \
    inception_v4_opt2.mlir \
    -o inception_v4_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter  inception_v4_quant_bf16.mlir \
    --tensor-in inception_v4_in_fp32.npz \
    --tensor-out inception_v4_out_bf16.npz \
    --dump-all-tensor=inception_v4_tensor_all_bf16.npz
npz_compare.py inception_v4_out_bf16.npz inception_v4_out_fp32.npz -v
# need to check torlerance later
npz_compare.py \
    inception_v4_tensor_all_bf16.npz \
    inception_v4_tensor_all_fp32.npz \
    --tolerance=0.99,0.99,0.88 -vvv

# VERDICT
echo $0 PASSED
