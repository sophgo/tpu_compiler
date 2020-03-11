#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/pytorch_alphapose/data/alphapose_threshold_table \
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

npz_compare.py \
    alphapose_tensor_all_int8.npz  \
    alphapose_tensor_all_fp32.npz \
    --op_info alphapose_op_info_int8_multiplier.csv \
    --dequant \
    --stats_int8_tensor \
    --save alpha.csv \
    --tolerance=0.9,0.9,-0.6 -vv

# VERDICT
echo $0 PASSED
