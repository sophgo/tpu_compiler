#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

CUSTOM_OP_PLUGIN_OPTION=""
if [[ ! -z $CUSTOM_OP_PLUGIN ]]; then
    CUSTOM_OP_PLUGIN_OPTION="--custom-op-plugin ${CUSTOM_OP_PLUGIN}"
fi

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
mlir-opt \
    --divide-ops-to-func \
    ${NET}_quant_int8_multiplier_tl_lw_memopt.mlir \
    -o ${NET}_quant_int8_multiplier_tl_lw_memopt_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    --cvi-set-chip ${SET_CHIP_NAME} \
    ${CUSTOM_OP_PLUGIN_OPTION}\
    --weight-file weight_int8_multiplier.bin \
    ${NET}_quant_int8_multiplier_tl_lw_memopt_func.mlir \
    -o ${NET}_int8_lw_memopt.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_lw_memopt.cvimodel \
    --batch-num $BATCH_SIZE \
    --set-chip ${SET_CHIP_NAME} \
    --output ${NET}_cmdbuf_out_all_int8_lw_memopt.npz

if [ $COMPARE_ALL -eq 1 ]; then
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_lw_memopt.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info ${NET}_op_info_int8_multiplier.csv
fi

# if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
#   mv ${NET}_int8_lw_memopt.cvimodel $CVIMODEL_REL_PATH
# fi

# VERDICT
echo $0 PASSED
