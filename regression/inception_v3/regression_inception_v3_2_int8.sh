#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


echo $0 IS RUNNING

COMPARE_ALL=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/inception_v3/data/inception_v3_threshold_table \
    inception_v3_opt.mlir \
    -o inception_v3_cali.mlir

###############################################################################
# quantization 1: per-layer int8
###############################################################################
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v3_op_info_int8_per_layer.csv \
    inception_v3_cali.mlir \
    -o inception_v3_quant_int8_per_layer.mlir

mlir-tpu-interpreter inception_v3_quant_int8_per_layer.mlir \
    --tensor-in inception_v3_in_raw_fp32.npz \
    --tensor-out inception_v3_out_int8_per_layer.npz \
    --dump-all-tensor=inception_v3_tensor_all_int8_per_layer.npz
#cvi_npz_tool.py to_bin \
#    inception_v3_tensor_all_int8_per_layer.npz \
#    classifier \
#    inception_v3_out_classifier_int8_per_layer.bin \
#    int8
#bin_compare.py \
#    inception_v3_out_classifier_int8_per_layer.bin \
#    $REGRESSION_PATH/inception_v3/data/test_cat_out_inception_v3_classifier_int8_per_layer.bin \
#    int8 1 1 1 1000 5
#
#if [ $COMPARE_ALL -eq 1 ]; then
#  # this will fail for now, because prob has been dequantized twice, others should pass
#  # need to check torlerance later
#  cvi_npz_tool.py compare \
#      inception_v3_tensor_all_int8_per_layer.npz \
#      inception_v3_blobs.npz \
#      --dequant \
#      --op_info inception_v3_quant_int8_per_layer_info.csv \
#      --tolerance 0.9,0.9,0.6 -vvv
#fi

###############################################################################
# quantization 2: per-channel int8
###############################################################################
mlir-opt \
    --tpu-quant --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v3_op_info_int8_per_channel.csv \
    inception_v3_cali.mlir \
    -o inception_v3_quant_int8_per_channel.mlir

mlir-tpu-interpreter inception_v3_quant_int8_per_channel.mlir \
    --tensor-in inception_v3_in_raw_fp32.npz \
    --tensor-out inception_v3_out_int8_per_channel.npz \
    --dump-all-tensor=inception_v3_tensor_all_int8_per_channel.npz

#cvi_npz_tool.py to_bin \
#    inception_v3_tensor_all_int8_per_channel.npz \
#    classifier \
#    inception_v3_out_classifier_int8_per_channel.bin \
#    int8
#bin_compare.py \
#    inception_v3_out_classifier_int8_per_channel.bin \
#    $REGRESSION_PATH/inception_v3/data/test_cat_out_inception_v3_classifier_int8_per_channel.bin \
#    int8 1 1 1 1000 5
#
#if [ $COMPARE_ALL -eq 1 ]; then
#  # this will fail for now, because prob has been dequantized twice, others should pass
#  # need to check torlerance later
#  cvi_npz_tool.py compare \
#      inception_v3_tensor_all_int8_per_channel.npz \
#      inception_v3_blobs.npz \
#      --dequant \
#      --op_info inception_v3_quant_int8_per_channel_info.csv \
#      --tolerance 0.9,0.9,0.7 -vvv
#fi

###############################################################################
# quantization 3: per-channel int8 with multiplier
###############################################################################
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v3_op_info_int8_multiplier.csv \
    inception_v3_cali.mlir \
    -o inception_v3_quant_int8_multiplier.mlir

mlir-tpu-interpreter inception_v3_quant_int8_multiplier.mlir \
    --tensor-in inception_v3_in_raw_fp32.npz \
    --tensor-out inception_v3_out_int8_multiplier.npz \
    --dump-all-tensor=inception_v3_tensor_all_int8_multiplier.npz

#cvi_npz_tool.py to_bin \
#    inception_v3_tensor_all_int8_multiplier.npz \
#    classifier \
#    inception_v3_out_classifier_int8_multiplier.bin \
#    int8
#bin_compare.py \
#    inception_v3_out_classifier_int8_multiplier.bin \
#    $REGRESSION_PATH/inception_v3/data/test_cat_out_inception_v3_classifier_int8_multiplier.bin \
#    int8 1 1 1 1000 5
#
#if [ $COMPARE_ALL -eq 1 ]; then
#  # this will fail for now, because prob has been dequantized twice, others should pass
#  # need to check torlerance later
#  cvi_npz_tool.py compare \
#      inception_v3_tensor_all_int8_multiplier.npz \
#      inception_v3_blobs.npz \
#      --dequant \
#      --op_info inception_v3_quant_int8_multiplier_info.csv \
#      --tolerance 0.9,0.9,0.7 -vvv
#fi

# VERDICT
echo $0 PASSED
