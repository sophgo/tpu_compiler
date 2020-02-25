#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

TENSOR_IN_FILE=./data/bmface_v3_in_fp32_scale.npz

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/bmface_v3/data/bmface-v3_cali1024_threshold_table \
    bmface-v3_opt.mlir \
    -o bmface-v3_cali.mlir

# apply post-calibration optimizations
# skip, bmface-v3 has no relu layer.

# only test quant int8 multiplier
# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --print-tpu-op-info \
    --tpu-op-info-filename bmface-v3_op_info_int8_multiplier.csv \
    bmface-v3_cali.mlir \
    -o bmface-v3_quant_int8_multiplier.mlir

mlir-tpu-interpreter bmface-v3_quant_int8_multiplier.mlir \
    --tensor-in $TENSOR_IN_FILE \
    --tensor-out bmface-v3_out_int8_multiplier.npz \
    --dump-all-tensor=bmface-v3_tensor_all_int8_multiplier.npz


# the result of the compare script is passed currently.
npz_compare.py \
    bmface-v3_tensor_all_int8_multiplier.npz \
    bmface-v3_tensor_all_fp32.npz \
    --op_info bmface-v3_op_info_int8_multiplier.csv \
    --dequant \
    --tolerance 0.9,0.9,0.6 -v


# VERDICT
echo $0 PASSED

