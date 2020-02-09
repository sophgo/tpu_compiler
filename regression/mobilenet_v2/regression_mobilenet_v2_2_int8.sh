#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# import calibration table
# relu-overwrite-backward is the default (20200209)
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/mobilenet_v2/data/mobilenet_v2_calibration_table \
    mobilenet_v2_opt.mlir \
    -o mobilenet_v2_cali.mlir

# for mobilenet_v2_calibration_table, fwd and bwd are the same
mlir-opt \
    --import-calibration-table \
    --enable-cali-overwrite-threshold-forward-relu=true \
    --enable-cali-overwrite-threshold-backward-relu=false \
    --calibration-table $REGRESSION_PATH/mobilenet_v2/data/mobilenet_v2_calibration_table \
    mobilenet_v2_opt.mlir \
    -o mobilenet_v2_cali_fwd.mlir

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    mobilenet_v2_cali.mlir \
    -o mobilenet_v2_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --tpu-op-info-filename mobilenet_v2_op_info_int8_per_layer.csv \
    mobilenet_v2_opt_post_cali.mlir \
    -o mobilenet_v2_quant_int8_per_layer.mlir

mlir-tpu-interpreter mobilenet_v2_quant_int8_per_layer.mlir \
    --tensor-in mobilenet_v2_in_fp32.npz \
    --tensor-out mobilenet_v2_out_int8_per_layer.npz \
    --dump-all-tensor=mobilenet_v2_tensor_all_int8_per_layer.npz

npz_to_bin.py \
    mobilenet_v2_tensor_all_int8_per_layer.npz \
    fc7 \
    mobilenet_v2_out_fc7_int8_per_layer.bin \
    int8
bin_compare.py \
    mobilenet_v2_out_fc7_int8_per_layer.bin \
    $REGRESSION_PATH/mobilenet_v2/data/test_cat_out_mobilenet_v2_fc7_int8_per_layer.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      mobilenet_v2_tensor_all_int8_per_layer.npz \
      mobilenet_v2_blobs.npz \
      --op_info mobilenet_v2_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.58,0.56,-0.03 -vvv
fi

# quantization 2: per-channel int8
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --print-tpu-op-info \
    --tpu-op-info-filename mobilenet_v2_op_info_int8_per_channel.csv \
    mobilenet_v2_opt_post_cali.mlir \
    -o mobilenet_v2_quant_int8_per_channel.mlir

mlir-tpu-interpreter mobilenet_v2_quant_int8_per_channel.mlir \
    --tensor-in mobilenet_v2_in_fp32.npz \
    --tensor-out mobilenet_v2_out_int8_per_channel.npz \
    --dump-all-tensor=mobilenet_v2_tensor_all_int8_per_channel.npz

npz_to_bin.py \
    mobilenet_v2_tensor_all_int8_per_channel.npz \
    fc7 \
    mobilenet_v2_out_fc7_int8_per_channel.bin \
    int8
bin_compare.py \
    mobilenet_v2_out_fc7_int8_per_channel.bin \
    $REGRESSION_PATH/mobilenet_v2/data/test_cat_out_mobilenet_v2_fc7_int8_per_channel.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      mobilenet_v2_tensor_all_int8_per_channel.npz \
      mobilenet_v2_blobs.npz \
      --op_info mobilenet_v2_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.9,0.89,0.57 -vvv
fi

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename mobilenet_v2_op_info_int8_multiplier.csv \
    mobilenet_v2_opt_post_cali.mlir \
    -o mobilenet_v2_quant_int8_multiplier.mlir

mlir-tpu-interpreter mobilenet_v2_quant_int8_multiplier.mlir \
    --tensor-in mobilenet_v2_in_fp32.npz \
    --tensor-out mobilenet_v2_out_int8_multiplier.npz \
    --dump-all-tensor=mobilenet_v2_tensor_all_int8_multiplier.npz

npz_to_bin.py \
    mobilenet_v2_tensor_all_int8_multiplier.npz \
    fc7 \
    mobilenet_v2_out_fc7_int8_multiplier.bin \
    int8
bin_compare.py \
    mobilenet_v2_out_fc7_int8_multiplier.bin \
    $REGRESSION_PATH/mobilenet_v2/data/test_cat_out_mobilenet_v2_fc7_int8_multiplier.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL -eq 1 ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      mobilenet_v2_tensor_all_int8_multiplier.npz \
      mobilenet_v2_blobs.npz \
      --op_info mobilenet_v2_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.9,0.9,0.57 -vvv
fi

# VERDICT
echo $0 PASSED
