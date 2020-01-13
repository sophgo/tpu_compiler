#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/resnet50/data/resnet50_calibration_table \
    resnet50_opt.mlir \
    -o resnet50_cali.mlir

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    resnet50_cali.mlir \
    -o resnet50_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    resnet50_opt_post_cali.mlir \
    -o resnet50_quant_int8_per_layer.mlir

mlir-tpu-interpreter resnet50_quant_int8_per_layer.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8_per_layer.npz \
    --dump-all-tensor=resnet50_tensor_all_int8_per_layer.npz

npz_to_bin.py \
    resnet50_tensor_all_int8_per_layer.npz \
    fc1000 \
    resnet50_out_fc1000_int8_per_layer.bin \
    int8
bin_compare.py \
    resnet50_out_fc1000_int8_per_layer.bin \
    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_per_layer.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      resnet50_tensor_all_int8_per_layer.npz \
      resnet50_blobs.npz \
      --dequant $REGRESSION_PATH/resnet50/data/resnet50_calibration_table \
      --tolerance 0.9,0.9,0.6 -vvv
fi

# quantization 2: per-channel int8
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    resnet50_opt_post_cali.mlir \
    -o resnet50_quant_int8_per_channel.mlir

mlir-tpu-interpreter resnet50_quant_int8_per_channel.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8_per_channel.npz \
    --dump-all-tensor=resnet50_tensor_all_int8_per_channel.npz

npz_to_bin.py \
    resnet50_tensor_all_int8_per_channel.npz \
    fc1000 \
    resnet50_out_fc1000_int8_per_channel.bin \
    int8
bin_compare.py \
    resnet50_out_fc1000_int8_per_channel.bin \
    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_per_channel.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      resnet50_tensor_all_int8_per_channel.npz \
      resnet50_blobs.npz \
      --dequant $REGRESSION_PATH/resnet50/data/resnet50_calibration_table \
      --tolerance 0.9,0.9,0.7 -vvv
fi

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    resnet50_opt_post_cali.mlir \
    -o resnet50_quant_int8_multiplier.mlir

mlir-tpu-interpreter resnet50_quant_int8_multiplier.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_out_int8_multiplier.npz \
    --dump-all-tensor=resnet50_tensor_all_int8_multiplier.npz

npz_to_bin.py \
    resnet50_tensor_all_int8_multiplier.npz \
    fc1000 \
    resnet50_out_fc1000_int8_multiplier.bin \
    int8
bin_compare.py \
    resnet50_out_fc1000_int8_multiplier.bin \
    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_multiplier.bin \
    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      resnet50_tensor_all_int8_multiplier.npz \
      resnet50_blobs.npz \
      --dequant $REGRESSION_PATH/resnet50/data/resnet50_calibration_table \
      --tolerance 0.9,0.9,0.7 -vvv
fi

# VERDICT
echo $0 PASSED
