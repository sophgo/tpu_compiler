#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


################################
# prepare bf16 input
################################
cvi_npz_tool.py to_bin vgg16_in_fp32.npz input vgg16_in_fp32.bin
bin_fp32_to_bf16.py \
    vgg16_in_fp32.bin \
    vgg16_in_bf16.bin

################################
# quantization
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    --assign-layer-id \
    vgg16_quant_bf16.mlir \
    -o vgg16_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    vgg16_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir vgg16_quant_bf16_addr.mlir \
    --output=vgg16_bf16.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input vgg16_in_fp32.npz \
    --model vgg16_bf16.cvimodel \
    --output vgg16_cmdbuf_out_all_bf16.npz

# compare all tensors
cvi_npz_tool.py compare \
    vgg16_cmdbuf_out_all_bf16.npz \
    vgg16_tensor_all_bf16.npz \
    --op_info vgg16_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vvv

# VERDICT
echo $0 PASSED
