#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


################################
# prepare bf16 input
################################
cvi_npz_tool.py to_bin ${NET}_in_fp32.npz input ${NET}_in_fp32.bin
bin_fp32_to_bf16.py \
    ${NET}_in_fp32.bin \
    ${NET}_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    ${NET}_quant_bf16.mlir \
    -o ${NET}_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    ${NET}_quant_bf16_tg.mlir \
    -o ${NET}_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir ${NET}_quant_bf16_addr.mlir \
    --output=${NET}_bf16.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_bf16.cvimodel \
    --output ${NET}_cmdbuf_out_all_bf16.npz

cvi_npz_tool.py extract \
    ${NET}_cmdbuf_out_all_bf16.npz \
    ${NET}_out_bf16_three_layer.npz \
    layer82-conv,layer94-conv,layer106-conv

cvi_npz_tool.py compare \
      ${NET}_out_bf16_three_layer.npz \
      ${NET}_tensor_all_bf16.npz \
      --op_info ${NET}_op_info_bf16_per_layer.csv