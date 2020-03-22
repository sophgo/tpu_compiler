#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# create int8 input
cvi_npz_tool.py to_bin \
    alphapose_tensor_all_int8.npz \
    input \
    alphapose_in_int8.bin \
    int8

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    alphapose_quant_int8_multiplier.mlir \
    -o alphapose_quant_int8_multiplier_tg.mlir

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
    alphapose_quant_int8_multiplier_tg.mlir \
    -o  alphapose_quant_int8_multiplier_cmdbuf.mlir

mlir-translate \
    alphapose_quant_int8_multiplier_cmdbuf.mlir \
     --mlir-to-cmdbuf \
     -o cmdbuf.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir alphapose_quant_int8_multiplier_cmdbuf.mlir \
    --output alphapose_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input alphapose_in_fp32.npz \
    --model alphapose_int8_multiplier.cvimodel \
    --output alphapose_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    alphapose_tensor_all_int8.npz \
    alphapose_cmdbuf_out_all_int8_multiplier.npz \
    --op_info alphapose_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
