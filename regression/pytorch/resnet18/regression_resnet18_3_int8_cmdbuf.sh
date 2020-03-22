#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"



#  Lower for quantization
mlir-opt \
    --tpu-lower \
    resnet18_quant_int8_multiplier.mlir \
    -o resnet18_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    resnet18_quant_int8_multiplier_tg.mlir \
    -o  resnet18_quant_int8_multiplier_cmdbuf.mlir

mlir-translate \
    resnet18_quant_int8_multiplier_cmdbuf.mlir \
     --mlir-to-cmdbuf \
     -o cmdbuf.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir resnet18_quant_int8_multiplier_cmdbuf.mlir \
    --output resnet18_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet18_in_fp32.npz \
    --model resnet18_int8_multiplier.cvimodel \
    --output resnet18_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    resnet18_tensor_all_int8.npz \
    resnet18_cmdbuf_out_all_int8_multiplier.npz \
    --op_info resnet18_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
