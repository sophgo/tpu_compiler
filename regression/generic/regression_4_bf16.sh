#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

if [ $DO_NOT_BF16_UNDER_182x -eq 1 ]; then
  exit 0
fi
if [ $DO_LG_WITH_BF16 -eq 0 ]; then
  exit 0
fi

mlir-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --tpu-quant --quant-full-bf16 \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_bf16.csv \
    ${NET}_opt_fp32.mlir \
    -o ${NET}_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter ${NET}_quant_bf16.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_bf16.npz \
    --dump-all-tensor=${NET}_tensor_all_bf16.npz

cvi_npz_tool.py compare \
    ${NET}_out_bf16.npz \
    ${NET}_out_fp32.npz \
    --tolerance $TOLERANCE_BF16 -vv

cvi_npz_tool.py compare \
    ${NET}_tensor_all_bf16.npz \
    ${NET}_tensor_all_fp32.npz \
    --op_info ${NET}_op_info_bf16.csv \
    --tolerance $TOLERANCE_BF16 -vv

$DIR/../mlir_to_cvimodel.sh \
   ${NET}_quant_bf16.mlir \
   ${NET}_bf16.cvimodel

# run cvimodel
model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_bf16.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_bf16.npz

# compare all tensors
cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_bf16.npz \
    ${NET}_tensor_all_bf16.npz \
    --op_info ${NET}_op_info_bf16.csv \
    --tolerance=$TOLERANCE_BF16_CMDBUF -vv

# VERDICT
echo $0 PASSED
