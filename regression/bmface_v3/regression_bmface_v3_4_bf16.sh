#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

TENSOR_IN_FILE=./data/bmface_v3_in_fp32_scale.npz

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    bmface_v3_opt.mlir \
    -o bmface_v3_opt2.mlir

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    bmface_v3_opt2.mlir \
    -o bmface_v3_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter bmface_v3_quant_bf16.mlir \
    --tensor-in $TENSOR_IN_FILE \
    --tensor-out bmface_v3_out_bf16.npz \
    --dump-all-tensor=bmface_v3_tensor_all_bf16.npz

#$PYTOOL_PATH/cvi_npz_tool.py compare bmface_v3_out_bf16.npz bmface_v3_out_fp32.npz -v

cvi_npz_tool.py compare \
    bmface_v3_tensor_all_bf16.npz \
    bmface_v3_tensor_all_fp32.npz \
    --op_info bmface_v3_op_info.csv \
    --tolerance 0.9,0.9,0.7 -vvv
    #--tolerance=0.99,0.99,0.95 -vvv

# VERDICT
echo $0 PASSED
