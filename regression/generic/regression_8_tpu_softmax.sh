#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

mlir-opt ${NET}_opt_fp32.mlir \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table ${CALI_TABLE} \
    -o ${NET}_cali.mlir

# add option --quant-bf16-softmax to enable tpu softmax
mlir-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant \
    --quant-bf16-softmax \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
    ${NET}_cali.mlir \
    -o ${NET}_quant_int8_multiplier_softmax.mlir

mlir-tpu-interpreter ${NET}_quant_int8_multiplier_softmax.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_int8_multiplier_softmax.npz \
    --dump-all-tensor=${NET}_tensor_all_int8_multiplier_softmax.npz

$DIR/../mlir_to_cvimodel.sh \
    ${NET}_quant_int8_multiplier_softmax.mlir \
    ${NET}_int8_multiplier_softmax.cvimodel

# run cvimodel
model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_multiplier_softmax.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_softmax.npz

# compare all tensors
cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_multiplier_softmax.npz \
    ${NET}_tensor_all_int8_multiplier_softmax.npz \
    --op_info ${NET}_op_info_int8_multiplier.csv


# VERDICT
echo $0 PASSED
