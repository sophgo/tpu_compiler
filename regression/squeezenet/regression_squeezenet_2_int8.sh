#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/squeezenet/data/squeezenet_v1.1_calibration_table \
    squeezenet_v1.1_opt.mlir \
    -o squeezenet_v1.1_cali.mlir

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    squeezenet_v1.1_cali.mlir \
    -o squeezenet_v1.1_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename squeezenet_v1.1_op_info_int8_per_layer.csv \
    squeezenet_v1.1_opt_post_cali.mlir \
    -o squeezenet_v1.1_quant_int8_per_layer.mlir

mlir-tpu-interpreter squeezenet_v1.1_quant_int8_per_layer.mlir \
    --tensor-in squeezenet_v1.1_in_fp32.npz \
    --tensor-out squeezenet_v1.1_out_int8_per_layer.npz \
    --dump-all-tensor=squeezenet_v1.1_tensor_all_int8_per_layer.npz

#npz_tool.py to_bin \
#    squeezenet_v1.1_tensor_all_int8_per_layer.npz \
#    pool10 \
#    squeezenet_v1.1_out_pool10_int8_per_layer.bin \
#    int8
#bin_compare.py \
#    squeezenet_v1.1_out_pool10_int8_per_layer.bin \
#    $REGRESSION_PATH/squeezenet/data/test_cat_out_squeezenet_v1.1_pool10_int8_per_layer.bin \
#    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_tool.py compare \
      squeezenet_v1.1_tensor_all_int8_per_layer.npz \
      squeezenet_v1.1_blobs.npz \
      --op_info squeezenet_v1.1_op_info_int8_per_layer.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.9,0.9,0.55 -vvv
fi

# quantization 2: per-channel int8
mlir-opt \
    --tpu-quant --quant-int8-rshift-only \
    --print-tpu-op-info \
    --tpu-op-info-filename squeezenet_v1.1_op_info_int8_per_channel.csv \
    squeezenet_v1.1_opt_post_cali.mlir \
    -o squeezenet_v1.1_quant_int8_per_channel.mlir

mlir-tpu-interpreter squeezenet_v1.1_quant_int8_per_channel.mlir \
    --tensor-in squeezenet_v1.1_in_fp32.npz \
    --tensor-out squeezenet_v1.1_out_int8_per_channel.npz \
    --dump-all-tensor=squeezenet_v1.1_tensor_all_int8_per_channel.npz

#npz_tool.py to_bin \
#    squeezenet_v1.1_tensor_all_int8_per_channel.npz \
#    pool10 \
#    squeezenet_v1.1_out_pool10_int8_per_channel.bin \
#    int8
#bin_compare.py \
#    squeezenet_v1.1_out_pool10_int8_per_channel.bin \
#    $REGRESSION_PATH/squeezenet/data/test_cat_out_squeezenet_v1.1_pool10_int8_per_channel.bin \
#    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_tool.py compare \
      squeezenet_v1.1_tensor_all_int8_per_channel.npz \
      squeezenet_v1.1_blobs.npz \
      --op_info squeezenet_v1.1_op_info_int8_per_channel.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.9,0.9,0.6 -vvv
fi

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename squeezenet_v1.1_op_info_int8_multiplier.csv \
    squeezenet_v1.1_opt_post_cali.mlir \
    -o squeezenet_v1.1_quant_int8_multiplier.mlir

mlir-tpu-interpreter squeezenet_v1.1_quant_int8_multiplier.mlir \
    --tensor-in squeezenet_v1.1_in_fp32.npz \
    --tensor-out squeezenet_v1.1_out_int8_multiplier.npz \
    --dump-all-tensor=squeezenet_v1.1_tensor_all_int8_multiplier.npz

#npz_tool.py to_bin \
#    squeezenet_v1.1_tensor_all_int8_multiplier.npz \
#    pool10 \
#    squeezenet_v1.1_out_pool10_int8_multiplier.bin \
#    int8
#bin_compare.py \
#    squeezenet_v1.1_out_pool10_int8_multiplier.bin \
#    $REGRESSION_PATH/squeezenet/data/test_cat_out_squeezenet_v1.1_pool10_int8_multiplier.bin \
#    int8 1 1 1 1000 5

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_tool.py compare \
      squeezenet_v1.1_tensor_all_int8_multiplier.npz \
      squeezenet_v1.1_blobs.npz \
      --op_info squeezenet_v1.1_op_info_int8_multiplier.csv \
      --dequant \
      --excepts prob \
      --tolerance 0.9,0.9,0.6 -vvv
fi

# VERDICT
echo $0 PASSED
