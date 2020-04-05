#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

REUSE_GLOBAL_MEM=${REUSE_GLOBAL_MEM:-0}

################################
# prepare int8 input
################################
cvi_npz_tool.py to_bin \
    resnet50_tensor_all_int8_multiplier.npz \
    data \
    resnet50_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# cvi_npz_tool.py to_bin resnet50_in_fp32.npz input resnet50_in_fp32.bin
# bin_fp32_to_int8.py \
#    resnet50_in_fp32.bin \
#    resnet50_in_int8.bin \
#    1.0 \
#    161.008057

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    resnet50_quant_int8_multiplier.mlir \
    -o resnet50_quant_int8_multiplier_tg.mlir

# function argument lower to MemRefType
mlir-opt \
    --debug \
    --convert-func-to-memref \
    resnet50_quant_int8_multiplier_tg.mlir \
    -o resnet50_quant_int8_multiplier_tg_memref.mlir

# op lower to MemRefType
mlir-opt \
    --debug \
    --convert-tg-op-to-memref \
    resnet50_quant_int8_multiplier_tg_memref.mlir \
    -o resnet50_quant_int8_multiplier_tg_op_memref.mlir

# memory space
mlir-opt \
    --debug \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref.csv \
    resnet50_quant_int8_multiplier_tg_op_memref.mlir \
    -o resnet50_quant_int8_multiplier_tg_op_memref_addr.mlir

# memory space w/ global memory reused
mlir-opt \
    --debug \
    --enable-reuse-global-mem=${REUSE_GLOBAL_MEM} \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
    resnet50_quant_int8_multiplier_tg_op_memref.mlir \
    -o resnet50_quant_int8_multiplier_tg_op_memref_addr_reuesed.mlir

# tg op back to TensorType
mlir-opt \
     --debug \
     --convert-tg-op-to-tensor \
     resnet50_quant_int8_multiplier_tg_op_memref_addr.mlir \
     -o resnet50_quant_int8_multiplier_tg_op_roundtrip.mlir

# function argument back to TensorType
mlir-opt \
    --debug \
    --convert-func-to-tensor \
    resnet50_quant_int8_multiplier_tg_op_roundtrip.mlir \
    -o resnet50_quant_int8_multiplier_tg_func_roundtrip.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet50_quant_int8_multiplier_tg_func_roundtrip.mlir \
    -o resnet50_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir resnet50_quant_int8_multiplier_addr.mlir \
    --output=resnet50_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet50_in_fp32.npz \
    --model resnet50_int8_multiplier.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    resnet50_cmdbuf_out_all_int8_multiplier.npz \
    resnet50_tensor_all_int8_multiplier.npz \
    --op_info resnet50_op_info_int8_multiplier.csv

#################
# Global memory reuse
#################
# memory space w/ global memory reuse
mlir-opt \
    --debug \
    --enable-reuse-global-mem=${REUSE_GLOBAL_MEM} \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
    resnet50_quant_int8_multiplier_tg_op_memref.mlir \
    -o resnet50_quant_int8_multiplier_tg_op_memref_addr_reused.mlir

# tg op back to TensorType
mlir-opt \
     --debug \
     --convert-tg-op-to-tensor \
     resnet50_quant_int8_multiplier_tg_op_memref_addr_reused.mlir \
     -o resnet50_quant_int8_multiplier_tg_op_roundtrip_reused.mlir

# function argument back to TensorType
mlir-opt \
    --debug \
    --convert-func-to-tensor \
    resnet50_quant_int8_multiplier_tg_op_roundtrip_reused.mlir \
    -o resnet50_quant_int8_multiplier_tg_func_roundtrip_reused.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_reused.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_reused.csv \
    resnet50_quant_int8_multiplier_tg_func_roundtrip_reused.mlir \
    -o resnet50_quant_int8_multiplier_addr_reused.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_int8_multiplier_addr_reused.mlir \
    -o cmdbuf_int8_multiplier_reused.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier_reused.bin \
    --weight weight_int8_multiplier.bin \
    --mlir resnet50_quant_int8_multiplier_addr_reused.mlir \
    --output=resnet50_int8_multiplier_reused.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet50_in_fp32.npz \
    --model resnet50_int8_multiplier_reused.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_multiplier_reused.npz

# compare all tensors
cvi_npz_tool.py compare \
    resnet50_cmdbuf_out_all_int8_multiplier_reused.npz \
    resnet50_tensor_all_int8_multiplier.npz \
    --op_info resnet50_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
