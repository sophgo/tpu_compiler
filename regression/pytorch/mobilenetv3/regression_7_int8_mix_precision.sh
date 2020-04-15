#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1

## import calibration table
##gdb --args \
##    --calibration-table ${NET}_preprocess_calibration_table \
##    --calibration-table tune_threshold_table \
mlir-opt \
    --import-calibration-table \
    --calibration-table ${NET}_preprocess_calibration_table \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir
##
#
## overwrite clip threshold to its parant and delete itself
### quantization: per-channel int8
##    --tpu-quant-clip \
##gdb --args \
mlir-opt \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
    ${NET}_cali.mlir \
    -o ${NET}_quant_int8_multiplier_0.mlir

mlir-opt \
    --tpu-quant \
    --quant-int8-mix-bf16-layers "316_Clip" \
    --quant-int8-mix-bf16-layers "354_Clip" \
    --quant-int8-mix-bf16-layers "372_Clip" \
    --quant-int8-mix-bf16-layers "391_Clip" \
    --quant-int8-mix-bf16-layers "402_Clip" \
    --quant-int8-mix-bf16-layers "410_Clip" \
    --quant-int8-mix-bf16-layers "420_Clip" \
    --quant-int8-mix-bf16-layers "428_Clip" \
    --quant-int8-mix-bf16-layers "428_Clip" \
    --quant-int8-mix-bf16-layers "439_Clip" \
    --quant-int8-mix-bf16-layers "447_Clip" \
    --quant-int8-mix-bf16-layers "458_Clip" \
    --quant-int8-mix-bf16-layers "466_Clip" \
    --quant-int8-mix-bf16-layers "477_Clip" \
    --quant-int8-mix-bf16-layers "485_Clip" \
    --quant-int8-mix-bf16-layers "493_Clip" \
    --quant-int8-mix-bf16-layers "500_Clip" \
    --quant-int8-mix-bf16-layers "510_Clip" \
    --quant-int8-mix-bf16-layers "518_Clip" \
    --quant-int8-mix-bf16-layers "526_Clip" \
    --quant-int8-mix-bf16-layers "533_Clip" \
    --quant-int8-mix-bf16-layers "544_Clip" \
    --quant-int8-mix-bf16-layers "552_Clip" \
    --quant-int8-mix-bf16-layers "560_Clip" \
    --quant-int8-mix-bf16-layers "567_Clip" \
    --quant-int8-mix-bf16-layers "577_Clip" \
    --quant-int8-mix-bf16-layers "585_Clip" \
    --quant-int8-mix-bf16-layers "593_Clip" \
    --quant-int8-mix-bf16-layers "600_Clip" \
    --quant-int8-mix-bf16-layers "611_Clip" \
    --quant-int8-mix-bf16-layers "619_Clip" \
    --quant-int8-mix-bf16-layers "627_Clip" \
    --quant-int8-mix-bf16-layers "634_Clip" \
    --quant-int8-mix-bf16-layers "645_Clip" \
    --quant-int8-mix-bf16-layers "653_Clip" \
    --quant-int8-mix-bf16-layers "313_BatchNormalization" \
    --quant-int8-mix-bf16-layers "315_Add" \
    --quant-int8-mix-bf16-layers "319_Mul" \
    --quant-int8-mix-bf16-layers "322_Relu" ${comment#54 passed} \
    --quant-int8-mix-bf16-layers "353_Add"  ${comment# 153 passed} \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
    ${NET}_quant_int8_multiplier_0.mlir \
    -o ${NET}_quant_int8_multiplier.mlir


# test mlir interpreter
#gdb --args \
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
