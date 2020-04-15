#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    ${NET}_quant_int8_multiplier.mlir \
    -o ${NET}_quant_int8_multiplier_tg.mlir

# FORCE do MemRefType cuz not back compatible
# function argument lower to MemRefType
mlir-opt \
    --convert-func-to-memref \
    ${NET}_quant_int8_multiplier_tg.mlir \
    -o ${NET}_quant_int8_multiplier_tg_opt_memref.mlir

# op lower to MemRefType
mlir-opt \
  --convert-tg-op-to-memref \
  ${NET}_quant_int8_multiplier_tg_opt_memref.mlir \
  -o ${NET}_quant_int8_multiplier_tg_opt_op_memref.mlir

# memory space w/ global memory reuse
mlir-opt \
    --enable-reuse-global-memory=false \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
    ${NET}_quant_int8_multiplier_tg_opt_op_memref.mlir \
    -o ${NET}_quant_int8_multiplier_tg_opt_op_memref_addr.mlir

# tg op back to TensorType
mlir-opt \
    --convert-tg-op-to-tensor \
    ${NET}_quant_int8_multiplier_tg_opt_op_memref_addr.mlir \
    -o ${NET}_quant_int8_multiplier_tg_opt_op_tensor_addr.mlir

# function argument back to TensorType
mlir-opt \
    --convert-func-to-tensor \
    ${NET}_quant_int8_multiplier_tg_opt_op_tensor_addr.mlir \
    -o ${NET}_quant_int8_multiplier_tg_opt_addr.mlir

# assign weight address & neuron address
#gdb --args \
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=${NET}_weight_map_int8_multiplier.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=${NET}_neuron_map_int8_multiplier.csv \
    --convert-cpu-op \
    ${NET}_quant_int8_multiplier_tg_opt_addr.mlir \
    -o ${NET}_quant_int8_multiplier_addr.mlir

mlir-translate \
  --mlir-to-cmdbuf \
  ${NET}_quant_int8_multiplier_addr.mlir \
  -o cmdbuf_int8_multiplier.bin

# generate cvimodel
build_cvimodel.py \
  --cmdbuf cmdbuf_int8_multiplier.bin \
  --weight weight_int8_multiplier.bin \
  --mlir ${NET}_quant_int8_multiplier_addr.mlir \
  --output=${NET}_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_multiplier.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8_multiplier.npz


 compare all tensors
#cvi_npz_tool.py compare \
#    ${NET}_tensor_all_int8.npz \
#    ${NET}_cmdbuf_out_all_int8_multiplier.npz \
#    --op_info ${NET}_op_info_int8_multiplier.csv \
#    -vvv


# compare fp32 only
cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_multiplier.npz \
    ${NET}_tensor_all_fp32.npz \
    --op_info ${NET}_op_info_int8_multiplier.csv \
    --dequant \
    --save ${NET}_stat.csv \
    --tolerance=0.7,0.3,0.7 -vv



# VERDICT
echo $0 PASSED
