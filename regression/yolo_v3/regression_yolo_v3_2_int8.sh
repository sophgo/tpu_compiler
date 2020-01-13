#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

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
      --dequant $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
      --tolerance 0.9,0.9,0.75 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_per_layer.npz \
      yolo_v3_blobs.npz \
      --dequant $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
      --tolerance 0.9,0.9,0.7 -vvv
fi

################################
# quantization 2: per-channel int8
################################

mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
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
      --dequant $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
      --tolerance 0.9,0.9,0.8 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_per_channel.npz \
      yolo_v3_blobs.npz \
      --dequant $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
      --tolerance 0.9,0.9,0.7 -vvv
fi

################################
# quantization 3: per-channel multiplier int8
################################
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
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
      --dequant $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
      --tolerance 0.9,0.9,0.8 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_multiplier.npz \
      yolo_v3_blobs.npz \
      --dequant $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
      --tolerance 0.9,0.9,0.7 -vvv
fi

# VERDICT
echo $0 PASSED
