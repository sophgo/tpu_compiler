#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

COMPARE_ALL=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/arcface/data/arcface_res50_calibration_table \
    arcface_res50_opt.mlir \
    -o arcface_res50_cali.mlir

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename arcface_res50_op_info_int8_multiplier.csv \
    arcface_res50_cali.mlir \
    -o arcface_res50_quant_int8_multiplier.mlir

mlir-tpu-interpreter arcface_res50_quant_int8_multiplier.mlir \
    --tensor-in arcface_res50_in_fp32.npz \
    --tensor-out arcface_res50_out_int8_multiplier.npz \
    --dump-all-tensor=arcface_res50_tensor_all_int8_multiplier.npz


# the result of the compare script is passed currently.
if [ $COMPARE_ALL -eq 1 ]; then
cvi_npz_tool.py compare \
    arcface_res50_tensor_all_int8_multiplier.npz \
    arcface_res50_tensor_all_fp32.npz \
    --op_info arcface_res50_op_info_int8_multiplier.csv \
    --dequant \
    --tolerance 0.59,0.59,-0.12 -v
fi
# VERDICT
echo $0 PASSED

