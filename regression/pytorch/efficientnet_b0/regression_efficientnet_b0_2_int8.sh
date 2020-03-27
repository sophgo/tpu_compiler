#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/pytorch/efficientnet_b0/data/efficientnet_b0_calibration_table \
    efficientnet_b0_opt.mlir \
    -o efficientnet_b0_cali.mlir

# quantization: per-channel int8
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename efficientnet_b0_op_info_int8_multiplier.csv \
    efficientnet_b0_cali.mlir \
    -o efficientnet_b0_quant_int8_multiplier.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet_b0_quant_int8_multiplier.mlir \
    --tensor-in efficientnet_b0_in_fp32.npz  \
    --tensor-out efficientnet_b0_out_int8.npz \
    --dump-all-tensor=efficientnet_b0_tensor_all_int8.npz

cvi_npz_tool.py compare \
    efficientnet_b0_tensor_all_int8.npz  \
    efficientnet_b0_tensor_all_fp32.npz \
    --op_info efficientnet_b0_op_info_int8_multiplier.csv \
    --dequant \
    --save efficientnet_b0_stat.csv \
    --tolerance=0.7,0.3,0 -vv

# VERDICT
echo $0 PASSED
