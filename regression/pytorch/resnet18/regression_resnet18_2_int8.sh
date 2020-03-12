#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../../envsetup.sh

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/pytorch/resnet18/data/resnet18_threshold_table \
    resnet18_opt.mlir \
    -o resnet18_cali.mlir

# quantization: per-channel int8
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename resnet18_op_info_int8_multiplier.csv \
    resnet18_cali.mlir \
    -o resnet18_quant_int8_multiplier.mlir

# test mlir interpreter
mlir-tpu-interpreter resnet18_quant_int8_multiplier.mlir \
    --tensor-in resnet18_in_fp32.npz  \
    --tensor-out resnet18_out_int8.npz \
    --dump-all-tensor=resnet18_tensor_all_int8.npz

npz_compare.py \
    resnet18_tensor_all_int8.npz  \
    resnet18_tensor_all_fp32.npz \
    --op_info resnet18_op_info_int8_multiplier.csv \
    --dequant \
    --stats_int8_tensor \
    --save alpha.csv \
    --tolerance=0.9,0.9,-0.6 -vv

# VERDICT
echo $0 PASSED
