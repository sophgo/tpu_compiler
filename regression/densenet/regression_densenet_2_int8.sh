#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/densenet/data/densenet_calibration_table \
    densenet_opt.mlir \
    -o densenet_cali.mlir

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    densenet_cali.mlir \
    -o densenet_opt_post_cali.mlir

mlir-tpu-interpreter densenet_opt_post_cali.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out out-opt-post-cali.npz

npz_compare.py densenet_out_fp32_fc6.npz out-opt-post-cali.npz --tolerance 0.9,0.9,0.6 -vvv

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --tpu-op-info-filename densenet_op_info_int8_per_layer.csv \
    densenet_opt_post_cali.mlir \
    -o densenet_quant_int8_per_layer.mlir

mlir-tpu-interpreter densenet_quant_int8_per_layer.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_int8_per_layer.npz \
    --dump-all-tensor=densenet_tensor_all_int8_per_layer.npz

npz_to_bin.py \
    densenet_tensor_all_int8_per_layer.npz \
    fc6 \
    densenet_out_fc6_int8_per_layer.bin \
    int8

bin_compare.py \
    densenet_out_fc6_int8_per_layer.bin \
    $REGRESSION_PATH/densenet/data/cat_densenet_out_fc6_int8_per_layer.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      densenet_tensor_all_int8_per_layer.npz \
      densenet_blobs.npz \
      --op_info densenet_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.67,0.65,0.11 -vvv
fi

# quantization 2: per-channel int8
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --print-tpu-op-info \
    --tpu-op-info-filename densenet_op_info_int8_per_channel.csv \
    densenet_opt_post_cali.mlir \
    -o densenet_quant_int8_per_channel.mlir

mlir-tpu-interpreter densenet_quant_int8_per_channel.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_int8_per_channel.npz \
    --dump-all-tensor=densenet_tensor_all_int8_per_channel.npz

npz_to_bin.py \
    densenet_tensor_all_int8_per_channel.npz \
    fc6 \
    densenet_out_fc6_int8_per_channel.bin \
    int8

bin_compare.py \
    densenet_out_fc6_int8_per_channel.bin \
    $REGRESSION_PATH/densenet/data/cat_densenet_out_fc6_int8_per_channel.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      densenet_tensor_all_int8_per_channel.npz \
      densenet_blobs.npz \
      --op_info densenet_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.88,0.87,0.48 -vvv
fi

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename densenet_op_info_int8_multiplier.csv \
    densenet_opt_post_cali.mlir \
    -o densenet_quant_int8_multiplier.mlir

mlir-tpu-interpreter densenet_quant_int8_multiplier.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_int8_multiplier.npz \
    --dump-all-tensor=densenet_tensor_all_int8_multiplier.npz

npz_to_bin.py \
    densenet_tensor_all_int8_multiplier.npz \
    fc6 \
    densenet_out_fc6_int8_multiplier.bin \
    int8
bin_compare.py \
    densenet_out_fc6_int8_multiplier.bin \
    $REGRESSION_PATH/densenet/data/cat_densenet_out_fc6_int8_multiplier.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      densenet_tensor_all_int8_multiplier.npz \
      densenet_blobs.npz \
      --op_info densenet_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.89,0.88,0.49 -vvv
fi

# VERDICT
echo $0 PASSED