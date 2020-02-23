#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/efficientnet-b0/data/efficientnet-b0_threshold_table \
    efficientnet-b0_opt.mlir \
    -o efficientnet-b0_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    efficientnet-b0_cali.mlir \
    -o efficientnet-b0_quant_int8_per_layer.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet-b0_quant_int8_per_layer.mlir \
    --tensor-in $REGRESSION_PATH/efficientnet-b0/data/efficientnet_in_fp32.npz  \
    --tensor-out efficientnet_out_int8.npz \
    --dump-all-tensor=efficientnet_tensor_all_int8.npz 
    
npz_compare.py \
    efficientnet_tensor_all_int8.npz  \
    efficientnet_tensor_all_fp32.npz \
    --op_info efficientnet-b0_op_info.csv \
    --dequant -v
# VERDICT
echo $0 PASSED
