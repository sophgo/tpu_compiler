#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1
COMPARE_OUTPUT_BIT_TRUE=0

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $CALI_TABLE \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir

###############################################################################
# quantization 1: per-layer int8
###############################################################################
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_per_layer.csv \
    ${NET}_cali.mlir \
    -o ${NET}_quant_int8_per_layer.mlir

mlir-tpu-interpreter ${NET}_quant_int8_per_layer.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_int8_per_layer.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_per_layer.npz

npz_to_bin.py \
    ${NET}_tensor_all_int8_per_layer.npz \
    ${OUTPUTS} \
    ${NET}_out_${OUTPUTS}_int8_per_layer.bin \
    int8
if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  bin_compare.py \
      ${NET}_out_${OUTPUTS}_int8_per_layer.bin \
      $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_per_layer.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      ${NET}_tensor_all_int8_per_layer.npz \
      ${NET}_blobs.npz \
      --op_info ${NET}_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance $TOLERANCE_PER_TENSOR -vv
fi

###############################################################################
# quantization 2: per-channel int8
###############################################################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_per_channel.csv \
    ${NET}_cali.mlir \
    -o ${NET}_quant_int8_per_channel.mlir

mlir-tpu-interpreter ${NET}_quant_int8_per_channel.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_int8_per_channel.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_per_channel.npz

npz_to_bin.py \
    ${NET}_tensor_all_int8_per_channel.npz \
    ${OUTPUTS} \
    ${NET}_out_${OUTPUTS}_int8_per_channel.bin \
    int8
if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  bin_compare.py \
      ${NET}_out_${OUTPUTS}_int8_per_channel.bin \
      $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_per_channel.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      ${NET}_tensor_all_int8_per_channel.npz \
      ${NET}_blobs.npz \
      --op_info ${NET}_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance $TOLERANCE_RSHIFT_ONLY -vv
fi

###############################################################################
# quantization 3: per-channel int8 with multiplier
###############################################################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
    ${NET}_cali.mlir \
    -o ${NET}_quant_int8_multiplier.mlir

mlir-tpu-interpreter ${NET}_quant_int8_multiplier.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_int8_multiplier.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_multiplier.npz

npz_to_bin.py \
    ${NET}_tensor_all_int8_multiplier.npz \
    ${OUTPUTS} \
    ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
    int8
if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
  bin_compare.py \
      ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
      $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_multiplier.bin \
      int8 1 1 1 1000 5
fi

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      ${NET}_tensor_all_int8_multiplier.npz \
      ${NET}_blobs.npz \
      --op_info ${NET}_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance $TOLERANCE_MULTIPLER -vv
fi

# VERDICT
echo $0 PASSED
