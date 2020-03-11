#!/bin/bash
set -e

# gen fake table
cat <<EOT > test_calibration_table
data 1.073760986
Y 1.073760986
EOT

# import calibration table
#gdb --args \
mlir-opt \
    --import-calibration-table \
    --calibration-table test_calibration_table \
    test_opt.mlir \
    -o test_cali.mlir

###############################################################################
# quantization 1: per-layer int8
###############################################################################
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename test_op_info_int8_per_layer.csv \
    test_cali.mlir \
    -o test_quant_int8_per_layer.mlir

mlir-tpu-interpreter test_quant_int8_per_layer.mlir \
    --tensor-in test_in_fp32.npz \
    --tensor-out test_out_int8_per_layer.npz \
    --dump-all-tensor=test_tensor_all_int8_per_layer.npz

###############################################################################
# quantization 3: per-channel int8 with multiplier
###############################################################################
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename test_op_info_int8_multiplier.csv \
    test_cali.mlir \
    -o test_quant_int8_multiplier.mlir

mlir-tpu-interpreter test_quant_int8_multiplier.mlir \
    --tensor-in test_in_fp32.npz \
    --tensor-out test_out_int8_multiplier.npz \
    --dump-all-tensor=test_tensor_all_int8_multiplier.npz
