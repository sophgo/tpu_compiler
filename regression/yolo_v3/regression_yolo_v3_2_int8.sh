#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# calibration
# python ../llvm/projects/mlir/externals/calibration_tool/run_calibration.py \
#     yolo_v3 yolo_v3_416_opt.mlir \
#     $DATA_PATH/input_coco_100.txt \
#     --input_num=100

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali.mlir

################################
# quantization 1: per-layer int8
################################
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --tpu-op-info-filename yolo_v3_op_info_int8_per_layer.csv \
    yolo_v3_416_cali.mlir \
    -o yolo_v3_416_quant_int8_per_layer.mlir

mlir-tpu-interpreter yolo_v3_416_quant_int8_per_layer.mlir \
    --tensor-in yolo_v3_in_fp32.npz \
    --tensor-out yolo_v3_out_dequant_int8_per_layer.npz \
    --dump-all-tensor=yolo_v3_tensor_all_int8_per_layer.npz

npz_extract.py \
    yolo_v3_tensor_all_int8_per_layer.npz \
    yolo_v3_out_int8_per_layer.npz \
    layer82-conv,layer94-conv,layer106-conv
npz_compare.py \
      yolo_v3_out_int8_per_layer.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --tolerance 0.9,0.85,0.75 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_per_layer.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.75,0.7,0.1 -vvv
fi

################################
# quantization 2: per-channel int8
################################

mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --print-tpu-op-info \
    --tpu-op-info-filename yolo_v3_op_info_int8_per_channel.csv \
    yolo_v3_416_cali.mlir \
    -o yolo_v3_416_quant_int8_per_channel.mlir

mlir-tpu-interpreter yolo_v3_416_quant_int8_per_channel.mlir \
    --tensor-in yolo_v3_in_fp32.npz \
    --tensor-out yolo_v3_out_dequant_int8_per_channel.npz \
    --dump-all-tensor=yolo_v3_tensor_all_int8_per_channel.npz

npz_extract.py \
    yolo_v3_tensor_all_int8_per_channel.npz \
    yolo_v3_out_int8_per_channel.npz \
    layer82-conv,layer94-conv,layer106-conv
npz_compare.py \
      yolo_v3_out_int8_per_channel.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_channel.csv \
      --dequant \
      --tolerance 0.9,0.9,0.8 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_per_channel.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_channel.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
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
    --tpu-op-info-filename yolo_v3_op_info_int8_multiplier.csv \
    yolo_v3_416_cali.mlir \
    -o yolo_v3_416_quant_int8_multiplier.mlir

mlir-tpu-interpreter yolo_v3_416_quant_int8_multiplier.mlir \
    --tensor-in yolo_v3_in_fp32.npz \
    --tensor-out yolo_v3_out_dequant_int8_multiplier.npz \
    --dump-all-tensor=yolo_v3_tensor_all_int8_multiplier.npz

npz_extract.py \
    yolo_v3_tensor_all_int8_multiplier.npz \
    yolo_v3_out_int8_multiplier.npz \
    layer82-conv,layer94-conv,layer106-conv
npz_compare.py \
      yolo_v3_out_int8_multiplier.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_multiplier.csv \
      --dequant \
      --tolerance 0.9,0.9,0.8 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_multiplier.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_multiplier.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.85,0.8,0.35 -vvv
fi

# VERDICT
echo $0 PASSED
