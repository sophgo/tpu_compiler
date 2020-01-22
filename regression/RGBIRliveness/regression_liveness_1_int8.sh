#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/RGBIRliveness/data/liveness_threshold_table \
    liveness_opt.mlir \
    -o liveness_cali.mlir

# apply post-calibration optimizations
# not applying --fuse-eltwise for now
mlir-opt \
    --fuse-relu \
    liveness_cali.mlir \
    -o liveness_opt_post_cali.mlir

# quantization: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename liveness_op_info_int8_multiplier.csv \
    liveness_opt_post_cali.mlir \
    -o liveness_quant_int8_multiplier.mlir

mlir-tpu-interpreter liveness_quant_int8_multiplier.mlir \
    --tensor-in $REGRESSION_PATH/RGBIRliveness/data/liveness_in_fp32.npz \
    --tensor-out liveness_out_int8_multiplier.npz \
    --dump-all-tensor=liveness_tensor_all_int8_multiplier.npz

npz_compare.py \
    liveness_out_int8_multiplier.npz \
    $REGRESSION_PATH/RGBIRliveness/data/liveness_out_int8_multiplier_fc2.npz -vvv

if [ $COMPARE_ALL ]; then
  # this will fail for now, because prob has been dequantized twice, others should pass
  npz_compare.py \
      liveness_tensor_all_int8_multiplier.npz \
      $REGRESSION_PATH/RGBIRliveness/data/liveness_int8_multiplier_blobs.npz
fi

# VERDICT
echo $0 PASSED
