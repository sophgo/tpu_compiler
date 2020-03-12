#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py resnet50_in_fp32.npz input resnet50_in_fp32.bin
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

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    resnet50_quant_bf16_tg.mlir \
    -o resnet50_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir resnet50_quant_bf16_addr.mlir \
    --output=resnet50_bf16.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet50_in_bf16.bin \
    --model resnet50_bf16.cvimodel \
    --output resnet50_cmdbuf_out_all_bf16.npz

# compare all tensors
npz_compare.py \
    resnet50_cmdbuf_out_all_bf16.npz \
    resnet50_tensor_all_bf16.npz \
    --op_info resnet50_op_info_bf16.csv \
    --tolerance=0.99,0.99,0.96 -vv

# VERDICT
echo $0 PASSED
