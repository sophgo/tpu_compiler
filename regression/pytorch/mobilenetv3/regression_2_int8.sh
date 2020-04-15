#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table ${NET}_preprocess_calibration_table \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir


# overwrite clip threshold to its parant and delete itself
mlir-opt \
    --tpu-quant-clip \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
    ${NET}_cali.mlir \
    -o ${NET}_quant_int8_multiplier.mlir

# test mlir interpreter
mlir-tpu-interpreter ${NET}_quant_int8_multiplier.mlir \
    --tensor-in ${NET}_in_fp32.npz  \
    --tensor-out ${NET}_out_int8.npz \
    --dump-all-tensor=${NET}_tensor_all_int8.npz

cvi_npz_tool.py compare \
    ${NET}_tensor_all_int8.npz  \
    ${NET}_tensor_all_fp32.npz \
    --op_info ${NET}_op_info_int8_multiplier.csv \
    --dequant \
    --save ${NET}_stat.csv \
    --tolerance=0.7,0.3,0.7 -vv

# VERDICT
echo $0 PASSED
