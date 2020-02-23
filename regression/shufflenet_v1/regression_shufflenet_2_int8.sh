#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1
COMPARE_OUTPUT_BIT_TRUE=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/shufflenet_v1/data/shufflenet_threshold_table \
    shufflenet_opt.mlir \
    -o shufflenet_cali.mlir

###############################################################################
# quantization 1: per-layer int8
###############################################################################
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --tpu-op-info-filename shufflenet_op_info_int8_per_layer.csv \
    shufflenet_cali.mlir \
    -o shufflenet_quant_int8_per_layer.mlir

mlir-tpu-interpreter shufflenet_quant_int8_per_layer.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_int8_per_layer.npz \
    --dump-all-tensor=shufflenet_tensor_all_int8_per_layer.npz

if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  npz_to_bin.py \
      shufflenet_tensor_all_int8_per_layer.npz \
      fc1000 \
      shufflenet_out_fc1000_int8_per_layer.bin \
      int8
  bin_compare.py \
      shufflenet_out_fc1000_int8_per_layer.bin \
      $REGRESSION_PATH/shufflenet_v1/data/test_cat_out_shufflenet_fc1000_int8_per_layer.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      shufflenet_tensor_all_int8_per_layer.npz \
      shufflenet_blobs.npz \
      --op_info shufflenet_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.9,0.9,0.6 -vvv
fi

###############################################################################
# quantization 2: per-channel int8
###############################################################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --print-tpu-op-info \
    --tpu-op-info-filename shufflenet_op_info_int8_per_channel.csv \
    shufflenet_cali.mlir \
    -o shufflenet_quant_int8_per_channel.mlir

mlir-tpu-interpreter shufflenet_quant_int8_per_channel.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_int8_per_channel.npz \
    --dump-all-tensor=shufflenet_tensor_all_int8_per_channel.npz

if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  npz_to_bin.py \
      shufflenet_tensor_all_int8_per_channel.npz \
      fc1000 \
      shufflenet_out_fc1000_int8_per_channel.bin \
      int8
  bin_compare.py \
      shufflenet_out_fc1000_int8_per_channel.bin \
      $REGRESSION_PATH/shufflenet_v1/data/test_cat_out_shufflenet_fc1000_int8_per_channel.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      shufflenet_tensor_all_int8_per_channel.npz \
      shufflenet_blobs.npz \
      --op_info shufflenet_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.9,0.9,0.7 -vvv
fi

###############################################################################
# quantization 3: per-channel int8 with multiplier
###############################################################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename shufflenet_op_info_int8_multiplier.csv \
    shufflenet_cali.mlir \
    -o shufflenet_quant_int8_multiplier.mlir

mlir-tpu-interpreter shufflenet_quant_int8_multiplier.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_int8_multiplier.npz \
    --dump-all-tensor=shufflenet_tensor_all_int8_multiplier.npz

if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  npz_to_bin.py \
      shufflenet_tensor_all_int8_multiplier.npz \
      fc1000 \
      shufflenet_out_fc1000_int8_multiplier.bin \
      int8
  bin_compare.py \
      shufflenet_out_fc1000_int8_multiplier.bin \
      $REGRESSION_PATH/shufflenet_v1/data/test_cat_out_shufflenet_fc1000_int8_multiplier.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      shufflenet_tensor_all_int8_multiplier.npz \
      shufflenet_blobs.npz \
      --op_info shufflenet_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.96,0.95,0.72 -vvv
fi

# VERDICT
echo $0 PASSED
