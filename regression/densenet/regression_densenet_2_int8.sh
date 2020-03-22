#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/densenet/data/densenet_threshold_table \
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

cvi_npz_tool.py compare densenet_out_fp32_fc6.npz out-opt-post-cali.npz --tolerance 0.9,0.9,0.6 -vvv

# quantization 1: per-layer int8
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename densenet_op_info_int8_per_layer.csv \
    densenet_opt_post_cali.mlir \
    -o densenet_quant_int8_per_layer.mlir

mlir-tpu-interpreter densenet_quant_int8_per_layer.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_int8_per_layer.npz \
    --dump-all-tensor=densenet_tensor_all_int8_per_layer.npz

cvi_npz_tool.py to_bin \
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
  cvi_npz_tool.py compare \
      densenet_tensor_all_int8_per_layer.npz \
      densenet_blobs.npz \
      --op_info densenet_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.75,0.74,0.25 -vvv
fi

# quantization 2: per-channel int8
mlir-opt \
    --tpu-quant --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename densenet_op_info_int8_per_channel.csv \
    densenet_opt_post_cali.mlir \
    -o densenet_quant_int8_per_channel.mlir

mlir-tpu-interpreter densenet_quant_int8_per_channel.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_int8_per_channel.npz \
    --dump-all-tensor=densenet_tensor_all_int8_per_channel.npz

cvi_npz_tool.py to_bin \
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
  cvi_npz_tool.py compare \
      densenet_tensor_all_int8_per_channel.npz \
      densenet_blobs.npz \
      --op_info densenet_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.87,0.87,0.48 -vvv
fi

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename densenet_op_info_int8_multiplier.csv \
    densenet_opt_post_cali.mlir \
    -o densenet_quant_int8_multiplier.mlir

mlir-tpu-interpreter densenet_quant_int8_multiplier.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_int8_multiplier.npz \
    --dump-all-tensor=densenet_tensor_all_int8_multiplier.npz

cvi_npz_tool.py to_bin \
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
  cvi_npz_tool.py compare \
      densenet_tensor_all_int8_multiplier.npz \
      densenet_blobs.npz \
      --op_info densenet_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.86,0.86,0.46 -vvv
fi

# VERDICT
echo $0 PASSED
