#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/vgg16/data/vgg16_calibration_table \
    vgg16_opt.mlir \
    -o vgg16_cali.mlir

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    vgg16_cali.mlir \
    -o vgg16_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename vgg16_op_info_int8_per_layer.csv \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_per_layer.mlir

mlir-tpu-interpreter vgg16_quant_int8_per_layer.mlir \
    --tensor-in vgg16_in_fp32.npz \
    --tensor-out vgg16_out_int8_per_layer.npz \
    --dump-all-tensor=vgg16_tensor_all_int8_per_layer.npz

npz_to_bin.py \
    vgg16_tensor_all_int8_per_layer.npz \
    fc8 \
    vgg16_out_fc8_int8_per_layer.bin \
    int8
bin_compare.py \
    vgg16_out_fc8_int8_per_layer.bin \
    $REGRESSION_PATH/vgg16/data/test_cat_out_vgg16_fc8_int8_per_layer.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      vgg16_tensor_all_int8_per_layer.npz \
      vgg16_blobs.npz \
      --op_info vgg16_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.98,0.97,0.46 -vvv
fi

# quantization 2: per-channel int8
mlir-opt \
    --tpu-quant --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename vgg16_op_info_int8_per_channel.csv \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_per_channel.mlir

mlir-tpu-interpreter vgg16_quant_int8_per_channel.mlir \
    --tensor-in vgg16_in_fp32.npz \
    --tensor-out vgg16_out_int8_per_channel.npz \
    --dump-all-tensor=vgg16_tensor_all_int8_per_channel.npz

npz_to_bin.py \
    vgg16_tensor_all_int8_per_channel.npz \
    fc8 \
    vgg16_out_fc8_int8_per_channel.bin \
    int8
bin_compare.py \
    vgg16_out_fc8_int8_per_channel.bin \
    $REGRESSION_PATH/vgg16/data/test_cat_out_vgg16_fc8_int8_per_channel.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      vgg16_tensor_all_int8_per_channel.npz \
      vgg16_blobs.npz \
      --op_info vgg16_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.98,0.98,0.46 -vvv
fi

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename vgg16_op_info_int8_multiplier.csv \
    vgg16_opt_post_cali.mlir \
    -o vgg16_quant_int8_multiplier.mlir

mlir-tpu-interpreter vgg16_quant_int8_multiplier.mlir \
    --tensor-in vgg16_in_fp32.npz \
    --tensor-out vgg16_out_int8_multiplier.npz \
    --dump-all-tensor=vgg16_tensor_all_int8_multiplier.npz

npz_to_bin.py \
    vgg16_tensor_all_int8_multiplier.npz \
    fc8 \
    vgg16_out_fc8_int8_multiplier.bin \
    int8
bin_compare.py \
    vgg16_out_fc8_int8_multiplier.bin \
    $REGRESSION_PATH/vgg16/data/test_cat_out_vgg16_fc8_int8_multiplier.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      vgg16_tensor_all_int8_multiplier.npz \
      vgg16_blobs.npz \
      --op_info vgg16_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.98,0.98,0.46 -vvv
fi

# VERDICT
echo $0 PASSED
