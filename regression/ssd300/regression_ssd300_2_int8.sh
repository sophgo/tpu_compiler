#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# calibration
# python ../llvm/projects/mlir/externals/calibration_tool/run_calibration.py \
#     ssd300 ssd300_opt2.mlir \
#     $DATA_PATH/input_coco_100.txt \
#     --input_num=100

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/ssd300/data/ssd300_threshold_table \
    ssd300_opt2.mlir \
    -o ssd300_cali.mlir

################################
# quantization 1: per-layer int8
################################
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info_int8_per_layer.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_per_layer.mlir

mlir-tpu-interpreter ssd300_quant_int8_per_layer.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_dequant_int8_per_layer.npz \
    --dump-all-tensor=ssd300_tensor_all_int8_per_layer.npz

npz_extract.py \
    ssd300_tensor_all_int8_per_layer.npz \
    ssd300_out_int8_per_layer.npz \
    detection_out
npz_compare.py \
      ssd300_out_int8_per_layer.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_layer.csv \
      --dequant \
      --tolerance 0.9,0.85,0.75 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_tensor_all_int8_per_layer.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_layer.csv \
      --dequant \
      --excepts detection_out \
      --tolerance 0.75,0.7,0.1 -vvv
fi

################################
# quantization 2: per-channel int8
################################

mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info_int8_per_channel.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_per_channel.mlir

mlir-tpu-interpreter ssd300_quant_int8_per_channel.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_dequant_int8_per_channel.npz \
    --dump-all-tensor=ssd300_tensor_all_int8_per_channel.npz

npz_extract.py \
    ssd300_tensor_all_int8_per_channel.npz \
    ssd300_out_int8_per_channel.npz \
    detection_out
npz_compare.py \
      ssd300_out_int8_per_channel.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_channel.csv \
      --dequant \
      --tolerance 0.9,0.9,0.8 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_tensor_all_int8_per_channel.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_per_channel.csv \
      --dequant \
      --excepts detection_out \
      --tolerance 0.85,0.8,0.30 -vvv
fi

################################
# quantization 3: per-channel multiplier int8
################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info_int8_multiplier.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_multiplier.mlir

mlir-tpu-interpreter ssd300_quant_int8_multiplier.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_dequant_int8_multiplier.npz \
    --dump-all-tensor=ssd300_v3_tensor_all_int8_multiplier.npz

npz_extract.py \
    ssd300_tensor_all_int8_multiplier.npz \
    ssd300_out_int8_multiplier.npz \
    detection_out
npz_compare.py \
      ssd300_out_int8_multiplier.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_multiplier.csv \
      --dequant \
      --tolerance 0.9,0.9,0.8 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      ssd300_tensor_all_int8_multiplier.npz \
      ssd300_blobs.npz \
      --op_info ssd300_op_info_int8_multiplier.csv \
      --dequant \
      --excepts detection_out \
      --tolerance 0.85,0.8,0.35 -vvv
fi

# VERDICT
echo $0 PASSED
