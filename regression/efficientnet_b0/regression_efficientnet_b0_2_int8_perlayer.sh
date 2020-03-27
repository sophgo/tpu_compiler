#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/efficientnet_b0/data/efficientnet_b0_calibration_table \
    efficientnet_b0_opt.mlir \
    -o efficientnet_b0_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename efficientnet_b0_op_info_int8_per_layer.csv \
    efficientnet_b0_cali.mlir \
    -o efficientnet_b0_quant_int8_per_layer.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet_b0_quant_int8_per_layer.mlir \
    --tensor-in efficientnet_in_fp32.npz  \
    --tensor-out efficientnet_out_int8.npz \
    --dump-all-tensor=efficientnet_tensor_all_int8.npz

cvi_npz_tool.py compare \
    efficientnet_tensor_all_int8.npz  \
    efficientnet_tensor_all_fp32.npz \
    --op_info efficientnet_b0_op_info_int8_per_layer.csv \
    --dequant -vv

# VERDICT
echo $0 PASSED
