#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

# assuming ${NET}_quant_int8_multiplier_tl_lw.mlir already exists
# assuming ${NET}_in_fp32.bin already exists

# function argument lower to MemRefType
mlir-opt \
    --convert-func-to-memref \
    ${NET}_quant_int8_multiplier_tl_lw.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_memref.mlir

# op lower to MemRefType
mlir-opt \
    --convert-tg-op-to-memref \
    ${NET}_quant_int8_multiplier_tl_lw_memref.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_op_memref.mlir

# memory space w/ global memory reuse
mlir-opt \
    --enable-reuse-global-memory=true \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
    ${NET}_quant_int8_multiplier_tl_lw_op_memref.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_op_memref_addr.mlir

# tg op back to TensorType
mlir-opt \
    --convert-tg-op-to-tensor \
    ${NET}_quant_int8_multiplier_tl_lw_op_memref_addr.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_op_tensor_addr.mlir

# function argument back to TensorType
mlir-opt \
    --convert-func-to-tensor \
    ${NET}_quant_int8_multiplier_tl_lw_op_tensor_addr.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_memopt.mlir

# generate cmdbuf
mlir-translate \
    ${NET}_quant_int8_multiplier_tl_lw_memopt.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf_lw_memopt.bin

# generate cvimodel
build_cvimodel.py \
    --cmdbuf cmdbuf_lw_memopt.bin \
    --weight weight_int8_multiplier.bin \
    --mlir ${NET}_quant_int8_multiplier_tl_lw.mlir \
    --output=${NET}_int8_lw_memopt.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_lw_memopt.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_lw_memopt.npz

if [ $COMPARE_ALL -eq 1 ]; then
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_lw_memopt.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info ${NET}_op_info_int8_multiplier.csv
fi

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  # mv ${NET}_int8_la.cvimodel $CVIMODEL_REL_PATH
  mv ${NET}_cmdbuf_out_all_int8_lw_memopt.cvimodel $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
