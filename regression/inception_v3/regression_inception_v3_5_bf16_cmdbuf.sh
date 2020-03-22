#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo $0 IS RUNNING

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    inception_v3_quant_bf16.mlir \
    -o inception_v3_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    inception_v3_quant_bf16_tg.mlir \
    -o inception_v3_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    inception_v3_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir inception_v3_quant_bf16_addr.mlir \
    --output=inception_v3_bf16.cvimodel

## run cmdbuf
model_runner \
    --dump-all-tensors \
    --input inception_v3_in_raw_fp32.npz \
    --model inception_v3_bf16.cvimodel \
    --output inception_v3_cmdbuf_out_all_bf16.npz

# compare all tensors
cvi_npz_tool.py compare \
    inception_v3_cmdbuf_out_all_bf16.npz \
    inception_v3_tensor_all_bf16.npz \
    --order neuron_map_bf16.csv \
    --tolerance=0.99,0.99,0.90 -vvv

# VERDICT
echo $0 PASSED
