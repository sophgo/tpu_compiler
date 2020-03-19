#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1
COMPARE_OUTPUT_BIT_TRUE=0

if [ $DO_CALIBRATION -eq 1 ]; then
  # Calibration
  # imagenet : --dataset $DATASET_PATH/imagenet/img_val_extracted
  # wider    : --dataset $DATASET_PATH/widerface/WIDER_val
  python $TPU_PYTHON_PATH/dataset_util/gen_dataset_img_list.py \
      --dataset $DATASET_PATH/imagenet/img_val_extracted \
      --count $CALIBRATION_IMAGE_COUNT \
      --output_img_list cali_list_imagenet.txt

  python $TPU_PYTHON_PATH/run_calibration.py \
    ${NET}_opt.mlir \
    cali_list_imagenet.txt \
    --output_file=${NET}_threshold_table \
    --net_input_dims $NET_INPUT_DIMS \
    --raw_scale $RAW_SCALE \
    --mean $MEAN \
    --input_scale $INPUT_SCALE \
    --input_num=$CALIBRATION_IMAGE_COUNT
else
  cp $CALI_TABLE ${NET}_threshold_table
fi

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table ${NET}_threshold_table \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir

###############################################################################
# quantization 1: per-layer int8
###############################################################################
if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then
  mlir-opt \
      --tpu-quant --quant-int8-per-tensor \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_per_tensor.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_int8_per_tensor.mlir

  mlir-tpu-interpreter ${NET}_quant_int8_per_tensor.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_per_tensor.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_per_tensor.npz

  npz_tool.py to_bin \
      ${NET}_tensor_all_int8_per_tensor.npz \
      ${OUTPUTS} \
      ${NET}_out_${OUTPUTS}_int8_per_tensor.bin \
      int8

  if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
    bin_compare.py \
        ${NET}_out_${OUTPUTS}_int8_per_tensor.bin \
        $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_per_tensor.bin \
        int8 ${BATCH_SIZE} 1 1 1000 5
  fi

  if [ $COMPARE_ALL -eq 1 ]; then
    # this will fail for now, because prob has been dequantized twice, others should pass
    npz_tool.py compare \
        ${NET}_tensor_all_int8_per_tensor.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_per_tensor.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance $TOLERANCE_INT8_PER_TENSOR -vv
  fi
fi

###############################################################################
# quantization 2: per-channel int8
###############################################################################
if [ $DO_QUANT_INT8_RFHIFT_ONLY -eq 1 ]; then

  mlir-opt \
      --tpu-quant --quant-int8-rshift-only \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_rshift_only.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_int8_rshift_only.mlir

  mlir-tpu-interpreter ${NET}_quant_int8_rshift_only.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_rshift_only.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_rshift_only.npz

  npz_tool.py to_bin \
      ${NET}_tensor_all_int8_rshift_only.npz \
      ${OUTPUTS} \
      ${NET}_out_${OUTPUTS}_int8_rshift_only.bin \
      int8

  if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
    bin_compare.py \
        ${NET}_out_${OUTPUTS}_int8_rshift_only.bin \
        $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_rshift_only.bin \
        int8 ${BATCH_SIZE} 1 1 1000 5
  fi

  if [ $COMPARE_ALL -eq 1 ]; then
    # this will fail for now, because prob has been dequantized twice, others should pass
    npz_tool.py compare \
        ${NET}_tensor_all_int8_rshift_only.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_rshift_only.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance $TOLERANCE_INT8_RSHIFT_ONLY -vv
  fi
fi

###############################################################################
# quantization 3: per-channel int8 with multiplier
###############################################################################
if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then

  mlir-opt \
      --tpu-quant \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_int8_multiplier.mlir

  mlir-tpu-interpreter ${NET}_quant_int8_multiplier.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_multiplier.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_multiplier.npz

  npz_tool.py to_bin \
      ${NET}_tensor_all_int8_multiplier.npz \
      ${OUTPUTS} \
      ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
      int8
  if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
    bin_compare.py \
        ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
        $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_multiplier.bin \
        int8 ${BATCH_SIZE} 1 1 1000 5
  fi

  if [ $COMPARE_ALL -eq 1 ]; then
    # this will fail for now, because prob has been dequantized twice, others should pass
    npz_tool.py compare \
        ${NET}_tensor_all_int8_multiplier.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_multiplier.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance $TOLERANCE_INT8_MULTIPLER -vv \
        --stats_int8_tensor
  fi
fi

# VERDICT
echo $0 PASSED
