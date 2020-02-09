#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# calibration
# python $PYTHON_TOOLS_PATH/dataset_util/gen_dataset_img_list.py \
#     --dataset $DATASET_PATH/coco/val2017 \
#     --count 100 \
#     --output_img_list cali_list_coco_100.txt
# python ../llvm/projects/mlir/externals/calibration_tool/run_calibration.py \
#     yolo_v3 \
#     yolo_v3_416_opt.mlir \
#     cali_list_coco_100.txt \
#     --input_num=100

# import calibration table
# relu-overwrite-backward is the default (20200209)
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table_autotune \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali.mlir

# autotune, relu-overwreite-forward
mlir-opt \
    --import-calibration-table \
    --enable-cali-overwrite-threshold-forward-relu=true \
    --enable-cali-overwrite-threshold-backward-relu=false \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table_autotune \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali_fwd.mlir

# autotune, no relu-overwrite
mlir-opt \
    --import-calibration-table \
    --enable-cali-overwrite-threshold-forward-relu=false \
    --enable-cali-overwrite-threshold-backward-relu=false \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table_autotune \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali_no_overwrite.mlir

# non-autotune, relu-overwrite-backward
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali_no_tune_bwd.mlir

# non-autotune, relu-overwrite-forward
mlir-opt \
    --import-calibration-table \
    --enable-cali-overwrite-threshold-forward-relu=true \
    --enable-cali-overwrite-threshold-backward-relu=false \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali_no_tune_fwd.mlir

# non-autotune, no relu-overwrite
mlir-opt \
    --import-calibration-table \
    --enable-cali-overwrite-threshold-forward-relu=false \
    --enable-cali-overwrite-threshold-backward-relu=false \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali_no_tune_no_overwrite.mlir

# non-autotune, concat-overwrite-backward
mlir-opt \
    --import-calibration-table \
    --enable-cali-overwrite-threshold-backward-concat \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali_concat_bwd.mlir

# non-autotune, concat-overwrite-max
mlir-opt \
    --import-calibration-table \
    --enable-cali-overwrite-threshold-max-concat \
    --calibration-table $REGRESSION_PATH/yolo_v3/data/yolo_v3_threshold_table \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_cali_concat_max.mlir

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
      --tolerance 0.98,0.93,0.84 -vvv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)

if [ $COMPARE_ALL  -eq 1]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_per_layer.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.90,0.88,0.51 -vvv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)
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
      --tolerance 0.99,0.95,0.87 -vvv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)

if [ $COMPARE_ALL  -eq 1]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_per_channel.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_channel.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.92,0.90,0.58 -vvv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)
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
    -o yolo_v3_416_quant_int8_multiplier.mlir \
    yolo_v3_416_cali.mlir

mlir-tpu-interpreter \
    --tensor-in yolo_v3_in_fp32.npz \
    --tensor-out yolo_v3_out_dequant_int8_multiplier.npz \
    --dump-all-tensor yolo_v3_tensor_all_int8_multiplier.npz \
    yolo_v3_416_quant_int8_multiplier.mlir

npz_extract.py \
    yolo_v3_tensor_all_int8_multiplier.npz \
    yolo_v3_out_int8_multiplier.npz \
    layer82-conv,layer94-conv,layer106-conv
npz_compare.py \
      yolo_v3_out_int8_multiplier.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_multiplier.csv \
      --dequant \
      --tolerance 0.99,0.95,0.87 -vvv

      # --tolerance 0.99,0.95,0.87 -vvv  # for autotune-no-overwrite (with leakyrelu both pos and neg quant)
      # --tolerance 0.99,0.95,0.87 -vvv  # for autotune-relu-overwrite-backward (with leakyrelu only neg quant)
      # --tolerance 0.99,0.95,0.85 -vvv  # for autotune-relu-overwrite-forward (with leakyrelu only neg quant)

      # --tolerance 0.98,0.90,0.81 -vvv  # for cali-no-tune-no-overwrite (with leakyrelu both pos and neg quant)
      # --tolerance 0.98,0.90,0.81 -vvv  # for cali-no-tune-relu-overwrite-backward (with leakyrelu only neg quant)
      # --tolerance 0.97,0.83,0.66 -vvv  # for cali-no-tune-relu-overwrite-forward (with leakyrelu only neg quant)

      # --tolerance 0.99,0.96,0.88 -vvv  # for autotune-relu-overwrite-backward (before leakyrelu quant)
      # --tolerance 0.99,0.97,0.88 -vvv  # for autotune-relu-overwrite-forward (before leakyrelu quant)

      # --tolerance 0.99,0.90,0.82 -vvv  # for cali-no-tune-relu-overwrite-backward (before leakyrelu quant)
      # --tolerance 0.97,0.87,0.71 -vvv  # for cali-no-tune-relu-overwrite-forward (before leakyrelu quant)

if [ $COMPARE_ALL  -eq 1]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_tensor_all_int8_multiplier.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_multiplier.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.93,0.92,0.61 -vvv

      # --tolerance 0.92,0.90,0.57 -vvv  # for autotune-no-overwrite (with leakyrelu both pos and neg quant)
      # --tolerance 0.93,0.92,0.61 -vvv  # for autotune-relu-overwrite-backward (with leakyrelu only neg quant)
      # --tolerance 0.92,0.91,0.59 -vvv  # for autotune-relu-overwrite-forward (with leakyrelu only neg quant)

      # --tolerance 0.83,0.81,0.38 -vvv  # for cali-no-tune-no-overwrite (with leakyrelu both pos and neg quant)
      # --tolerance 0.83,0.81,0.36 -vvv  # for cali-no-tune-relu-overwrite-backward (with leakyrelu only neg quant)
      # --tolerance 0.81,0.75,0.26 -vvv  # for cali-no-tune-relu-overwrite-forward (with leakyrelu only neg quant)

      # --tolerance 0.93,0.93,0.64 -vvv  # for autotune-relu-overwrite-backward (before leakyrelu quant)
      # --tolerance 0.95,0.93,0.65 -vvv  # for autotune-relu-overwrite-forward (before leakyrelu quant)

      # --tolerance 0.83,0.81,0.36 -vvv  # for cali-no-tune-relu-overwrite-backward (before leakyrelu quant)
      # --tolerance 0.84,0.79,0.33 -vvv  # for cali-no-tune-relu-overwrite-forward (before leakyrelu quant)
fi

# VERDICT
echo $0 PASSED
