#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/pytorch/alphapose/data/alphapose_threshold_table \
    alphapose_opt.mlir \
    -o alphapose_cali.mlir

# quantization: per-channel int8
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename alphapose_op_info_int8_multiplier.csv \
    alphapose_cali.mlir \
    -o alphapose_quant_int8_multiplier.mlir

# test mlir interpreter
mlir-tpu-interpreter alphapose_quant_int8_multiplier.mlir \
    --tensor-in alphapose_in_fp32.npz  \
    --tensor-out alphapose_out_int8.npz \
    --dump-all-tensor=alphapose_tensor_all_int8.npz

cvi_npz_tool.py compare \
    alphapose_tensor_all_int8.npz  \
    alphapose_tensor_all_fp32.npz \
    --op_info alphapose_op_info_int8_multiplier.csv \
    --dequant \
    --stats_int8_tensor \
    --save alpha.csv \
    --tolerance=0.89,0.88,0.5 -vv

# VERDICT
echo $0 PASSED
