#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


################################
# prepare bf16 input
################################
cvi_npz_tool.py to_bin squeezenet_in_fp32.npz data squeezenet_in_fp32.bin
bin_fp32_to_bf16.py \
    squeezenet_in_fp32.bin \
    squeezenet_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower --reorder-op \
    squeezenet_quant_bf16.mlir \
    -o squeezenet_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    squeezenet_quant_bf16_tg.mlir \
    -o squeezenet_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    squeezenet_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir squeezenet_quant_bf16_addr.mlir \
    --output=squeezenet_bf16.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    squeezenet_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    squeezenet_cmdbuf_out_all_bf16.bin \
#    32921552 0 32921552 1
model_runner \
    --dump-all-tensors \
    --input squeezenet_in_fp32.npz \
    --model squeezenet_bf16.cvimodel \
    --output squeezenet_cmdbuf_out_all_bf16.npz

# compare all tensors
cvi_npz_tool.py compare \
    squeezenet_cmdbuf_out_all_bf16.npz \
    squeezenet_tensor_all_bf16.npz \
    --op_info squeezenet_op_info.csv \
    --tolerance=0.99,0.99,0.93 -vv

# VERDICT
echo $0 PASSED
