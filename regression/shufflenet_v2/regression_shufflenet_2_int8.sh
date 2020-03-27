#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/shufflenet_v2/data/shufflenet_v2_calibration_table \
    shufflenet_opt.mlir \
    -o shufflenet_cali.mlir

###############################################################################
# quantization 1: per-layer int8
###############################################################################
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename shufflenet_op_info_int8_per_layer.csv \
    shufflenet_cali.mlir \
    -o shufflenet_quant_int8_per_layer.mlir

mlir-tpu-interpreter shufflenet_quant_int8_per_layer.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_int8_per_layer.npz \
    --dump-all-tensor=shufflenet_tensor_all_int8_per_layer.npz

cvi_npz_tool.py compare \
    shufflenet_tensor_all_int8_per_layer.npz \
    shufflenet_blobs.npz \
    --op_info shufflenet_op_info_int8_per_layer.csv \
    --dequant \
    --tolerance 0.91,0.91,0.57 -vv

###############################################################################
# quantization 2: per-channel int8
###############################################################################
mlir-opt \
    --tpu-quant --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename shufflenet_op_info_int8_per_channel.csv \
    shufflenet_cali.mlir \
    -o shufflenet_quant_int8_per_channel.mlir

mlir-tpu-interpreter shufflenet_quant_int8_per_channel.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_int8_per_channel.npz \
    --dump-all-tensor=shufflenet_tensor_all_int8_per_channel.npz

cvi_npz_tool.py compare \
    shufflenet_tensor_all_int8_per_channel.npz \
    shufflenet_blobs.npz \
    --op_info shufflenet_op_info_int8_per_channel.csv \
    --dequant \
    --tolerance 0.94,0.94,0.67 -vv

###############################################################################
# quantization 3: per-channel int8 with multiplier
###############################################################################
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename shufflenet_op_info_int8_multiplier.csv \
    shufflenet_cali.mlir \
    -o shufflenet_quant_int8_multiplier.mlir

mlir-tpu-interpreter shufflenet_quant_int8_multiplier.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_int8_multiplier.npz \
    --dump-all-tensor=shufflenet_tensor_all_int8_multiplier.npz

cvi_npz_tool.py compare \
    shufflenet_tensor_all_int8_multiplier.npz \
    shufflenet_blobs.npz \
    --op_info shufflenet_op_info_int8_multiplier.csv \
    --dequant \
    --tolerance 0.95,0.95,0.67 -vv

# VERDICT
echo $0 PASSED
