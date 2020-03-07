#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/liveness/data/liveness_threshold_table \
    liveness_opt.mlir \
    -o liveness_cali.mlir

# quantization: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename liveness_op_info_int8_multiplier.csv \
    liveness_cali.mlir \
    -o liveness_quant_int8_multiplier.mlir

mlir-tpu-interpreter liveness_quant_int8_multiplier.mlir \
    --tensor-in liveness_in_fp32.npz \
    --tensor-out liveness_out_int8_multiplier.npz \
    --dump-all-tensor=liveness_tensor_all_int8_multiplier.npz

npz_compare.py \
    liveness_tensor_all_int8_multiplier.npz \
    liveness_tensor_all_fp32.npz \
    --op_info liveness_op_info_int8_multiplier.csv \
    --dequant \
    --tolerance 0.9,0.9,0.7 -v

# VERDICT
echo $0 PASSED
