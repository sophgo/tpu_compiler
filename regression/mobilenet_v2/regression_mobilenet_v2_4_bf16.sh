#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh


# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    mobilenet_v2_opt.mlir \
    -o mobilenet_v2_opt2.mlir

# quantization
mlir-opt \
    --quant-bf16 \
    mobilenet_v2_opt2.mlir \
    -o mobilenet_v2_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter mobilenet_v2_quant_bf16.mlir \
    --tensor-in mobilenet_v2_in_fp32.npz \
    --tensor-out mobilenet_v2_out_bf16.npz \
    --dump-all-tensor=mobilenet_v2_tensor_all_bf16.npz
npz_compare.py mobilenet_v2_out_bf16.npz mobilenet_v2_out_fp32.npz -v
npz_compare.py \
    mobilenet_v2_tensor_all_bf16.npz \
    mobilenet_v2_tensor_all_fp32.npz \
    --op_info mobilenet_v2_op_info.csv \
    --tolerance=0.99,0.99,0.93 -vvv

# VERDICT
echo $0 PASSED
