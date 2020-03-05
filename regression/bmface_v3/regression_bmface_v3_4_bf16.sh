#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

TENSOR_IN_FILE=./data/bmface_v3_in_fp32_scale.npz

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    bmface-v3_opt.mlir \
    -o bmface-v3_opt2.mlir

# quantization
mlir-opt \
    --quant-bf16 \
    bmface-v3_opt2.mlir \
    -o bmface-v3_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter bmface-v3_quant_bf16.mlir \
    --tensor-in $TENSOR_IN_FILE \
    --tensor-out bmface-v3_out_bf16.npz \
    --dump-all-tensor=bmface-v3_tensor_all_bf16.npz

#$PYTOOL_PATH/npz_compare.py bmface-v3_out_bf16.npz bmface-v3_out_fp32.npz -v

npz_compare.py \
    bmface-v3_tensor_all_bf16.npz \
    bmface-v3_tensor_all_fp32.npz \
    --op_info bmface-v3_op_info.csv \
    --tolerance 0.9,0.9,0.7 -vvv
    #--tolerance=0.99,0.99,0.95 -vvv

# VERDICT
echo $0 PASSED
