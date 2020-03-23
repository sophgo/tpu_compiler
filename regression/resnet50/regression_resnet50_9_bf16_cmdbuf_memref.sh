#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


################################
# prepare bf16 input
################################
cvi_npz_tool.py to_bin resnet50_in_fp32.npz input resnet50_in_fp32.bin
bin_fp32_to_bf16.py \
    resnet50_in_fp32.bin \
    resnet50_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    resnet50_quant_bf16.mlir \
    -o resnet50_quant_bf16_tg.mlir

# function argument lower to MemRefType
mlir-opt \
    --debug \
    --convert-func-to-memref \
    resnet50_quant_bf16_tg.mlir \
    -o resnet50_quant_bf16_tg_memref.mlir

# op lower to MemRefType
mlir-opt \
    --debug \
    --convert-tg-op-to-memref \
    resnet50_quant_bf16_tg_memref.mlir \
    -o resnet50_quant_bf16_tg_op_memref.mlir

# memory space
mlir-opt \
    --debug \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref.csv \
    resnet50_quant_bf16_tg_op_memref.mlir \
    -o resnet50_quant_bf16_tg_op_memref_addr.mlir \

# tg op back to TensorType
mlir-opt \
     --debug \
     --convert-tg-op-to-tensor \
     resnet50_quant_bf16_tg_op_memref_addr.mlir \
     -o resnet50_quant_bf16_tg_op_roundtrip.mlir

# function argument back to TensorType
mlir-opt \
    --debug \
    --convert-func-to-tensor \
    resnet50_quant_bf16_tg_op_roundtrip.mlir \
    -o resnet50_quant_bf16_tg_func_roundtrip.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    resnet50_quant_bf16_tg_func_roundtrip.mlir \
    -o resnet50_quant_bf16_addr_roundtrip.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_bf16_addr_roundtrip.mlir \
    -o cmdbuf_bf16_roundtrip.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16_roundtrip.bin \
    --weight weight_bf16.bin \
    --mlir resnet50_quant_bf16_addr_roundtrip.mlir \
    --output=resnet50_bf16_roundtrip.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet50_in_fp32.npz \
    --model resnet50_bf16_roundtrip.cvimodel \
    --output resnet50_cmdbuf_out_all_bf16_roundtrip.npz

# compare all tensors
cvi_npz_tool.py compare \
    resnet50_cmdbuf_out_all_bf16_roundtrip.npz \
    resnet50_tensor_all_bf16.npz \
    --op_info resnet50_op_info_bf16.csv \
    --tolerance=0.99,0.99,0.96 -vv

#################
# Neuron map recycle
#################
# memory space w/ neuron recycle
mlir-opt \
    --debug \
    --enable-tpu-neuron-map-recyle-memref=1 \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref_recycle.csv \
    resnet50_quant_bf16_tg_op_memref.mlir \
    -o resnet50_quant_bf16_tg_op_memref_addr_recycle.mlir

# tg op back to TensorType
mlir-opt \
     --debug \
     --convert-tg-op-to-tensor \
     resnet50_quant_bf16_tg_op_memref_addr_recycle.mlir \
     -o resnet50_quant_bf16_tg_op_roundtrip_recycle.mlir

# function argument back to TensorType
mlir-opt \
    --debug \
    --convert-func-to-tensor \
    resnet50_quant_bf16_tg_op_roundtrip_recycle.mlir \
    -o resnet50_quant_bf16_tg_func_roundtrip_recycle.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16_recycle.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16_recycle.csv \
    resnet50_quant_bf16_tg_func_roundtrip_recycle.mlir \
    -o resnet50_quant_bf16_addr_roundtrip_recycle.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_bf16_addr_roundtrip_recycle.mlir \
    -o cmdbuf_bf16_roundtrip_recycle.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16_roundtrip_recycle.bin \
    --weight weight_bf16.bin \
    --mlir resnet50_quant_bf16_addr_roundtrip_recycle.mlir \
    --output=resnet50_bf16_roundtrip_recycle.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet50_in_fp32.npz \
    --model resnet50_bf16_roundtrip_recycle.cvimodel \
    --output resnet50_cmdbuf_out_all_bf16_roundtrip_recycle.npz

# compare all tensors
cvi_npz_tool.py compare \
    resnet50_cmdbuf_out_all_bf16_roundtrip_recycle.npz \
    resnet50_tensor_all_bf16.npz \
    --op_info resnet50_op_info_bf16.csv \
    --tolerance=0.99,0.99,0.96 -vv

# VERDICT
echo $0 PASSED
