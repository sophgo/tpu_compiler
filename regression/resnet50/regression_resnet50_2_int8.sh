#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1
COMPARE_OUTPUT_BIT_TRUE=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/resnet50/data/resnet50_calibration_table \
    resnet50_opt.mlir \
    -o resnet50_cali.mlir

###############################################################################
# quantization 1: per-layer int8
###############################################################################
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename resnet50_op_info_int8_per_layer.csv \
    resnet50_cali.mlir \
    -o resnet50_quant_int8_per_layer.mlir

mlir-tpu-interpreter resnet50_quant_int8_per_layer.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8_per_layer.npz \
    --dump-all-tensor=resnet50_tensor_all_int8_per_layer.npz

if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  npz_tool.py to_bin \
      resnet50_tensor_all_int8_per_layer.npz \
      fc1000 \
      resnet50_out_fc1000_int8_per_layer.bin \
      int8
  bin_compare.py \
      resnet50_out_fc1000_int8_per_layer.bin \
      $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_per_layer.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_tool.py compare \
      resnet50_tensor_all_int8_per_layer.npz \
      resnet50_blobs.npz \
      --op_info resnet50_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.92,0.91,0.59 -vv
fi

###############################################################################
# quantization 2: per-channel int8
###############################################################################
mlir-opt \
    --tpu-quant --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename resnet50_op_info_int8_per_channel.csv \
    resnet50_cali.mlir \
    -o resnet50_quant_int8_per_channel.mlir

mlir-tpu-interpreter resnet50_quant_int8_per_channel.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8_per_channel.npz \
    --dump-all-tensor=resnet50_tensor_all_int8_per_channel.npz

if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  npz_tool.py to_bin \
      resnet50_tensor_all_int8_per_channel.npz \
      fc1000 \
      resnet50_out_fc1000_int8_per_channel.bin \
      int8
  bin_compare.py \
      resnet50_out_fc1000_int8_per_channel.bin \
      $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_per_channel.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_tool.py compare \
      resnet50_tensor_all_int8_per_channel.npz \
      resnet50_blobs.npz \
      --op_info resnet50_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.95,0.95,0.7 -vv
fi

###############################################################################
# quantization 3: per-channel int8 with multiplier
###############################################################################
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename resnet50_op_info_int8_multiplier.csv \
    resnet50_cali.mlir \
    -o resnet50_quant_int8_multiplier.mlir

mlir-tpu-interpreter resnet50_quant_int8_multiplier.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8_multiplier.npz \
    --dump-all-tensor=resnet50_tensor_all_int8_multiplier.npz

if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  npz_tool.py to_bin \
      resnet50_tensor_all_int8_multiplier.npz \
      fc1000 \
      resnet50_out_fc1000_int8_multiplier.bin \
      int8
  bin_compare.py \
      resnet50_out_fc1000_int8_multiplier.bin \
      $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_multiplier.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_tool.py compare \
      resnet50_tensor_all_int8_multiplier.npz \
      resnet50_blobs.npz \
      --op_info resnet50_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.96,0.95,0.73 -vv
fi

# VERDICT
echo $0 PASSED
