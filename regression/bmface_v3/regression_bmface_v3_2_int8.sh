#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/bmface_v3/data/bmface-v3_cali1024_threshold_table \
    bmface_v3_opt.mlir \
    -o bmface_v3_cali.mlir

# quantization 3: per-channel int8 with multiplier
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename bmface_v3_op_info_int8_multiplier.csv \
    bmface_v3_cali.mlir \
    -o bmface_v3_quant_int8_multiplier.mlir

mlir-tpu-interpreter bmface_v3_quant_int8_multiplier.mlir \
    --tensor-in bmface_v3_in_fp32.npz \
    --tensor-out bmface_v3_out_int8_multiplier.npz \
    --dump-all-tensor=bmface_v3_tensor_all_int8_multiplier.npz


# the result of the compare script is passed currently.
npz_compare.py \
    bmface_v3_tensor_all_int8_multiplier.npz \
    bmface_v3_tensor_all_fp32.npz \
    --op_info bmface_v3_op_info_int8_multiplier.csv \
    --dequant \
    --tolerance 0.9,0.9,0.6 -v

# VERDICT
echo $0 PASSED

