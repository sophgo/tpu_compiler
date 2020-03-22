#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# create int8 input
cvi_npz_tool.py to_bin \
    efficientnet_b0_tensor_all_int8.npz \
    input \
    efficientnet_b0_in_int8.bin \
    int8

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    efficientnet_b0_quant_int8_multiplier.mlir \
    -o efficientnet_b0_quant_int8_multiplier_tg.mlir

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
    efficientnet_b0_quant_int8_multiplier_tg.mlir \
    -o  efficientnet_b0_quant_int8_multiplier_cmdbuf.mlir

mlir-translate \
    efficientnet_b0_quant_int8_multiplier_cmdbuf.mlir \
     --mlir-to-cmdbuf \
     -o cmdbuf.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir efficientnet_b0_quant_int8_multiplier_cmdbuf.mlir \
    --output efficientnet_b0_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input efficientnet_b0_in_fp32.npz \
    --model efficientnet_b0_int8_multiplier.cvimodel \
    --output efficientnet_b0_cmdbuf_out_all_int8_multiplier.npz


# compare all tensors
cvi_npz_tool.py compare \
    efficientnet_b0_tensor_all_int8.npz \
    efficientnet_b0_cmdbuf_out_all_int8_multiplier.npz \
    --op_info efficientnet_b0_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
