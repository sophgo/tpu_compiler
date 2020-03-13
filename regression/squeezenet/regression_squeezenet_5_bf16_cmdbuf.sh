#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py squeezenet_v1.1_in_fp32.npz data squeezenet_v1.1_in_fp32.bin
bin_fp32_to_bf16.py \
    squeezenet_v1.1_in_fp32.bin \
    squeezenet_v1.1_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    squeezenet_v1.1_quant_bf16.mlir \
    -o squeezenet_v1.1_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    squeezenet_v1.1_quant_bf16_tg.mlir \
    -o squeezenet_v1.1_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    squeezenet_v1.1_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir squeezenet_v1.1_quant_bf16_addr.mlir \
    --output=squeezenet_v1.1_bf16.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    squeezenet_v1.1_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    squeezenet_v1.1_cmdbuf_out_all_bf16.bin \
#    32921552 0 32921552 1
model_runner \
    --dump-all-tensors \
    --input squeezenet_v1.1_in_fp32.npz \
    --model squeezenet_v1.1_bf16.cvimodel \
    --output squeezenet_v1.1_cmdbuf_out_all_bf16.npz

# compare all tensors
npz_compare.py \
    squeezenet_v1.1_cmdbuf_out_all_bf16.npz \
    squeezenet_v1.1_tensor_all_bf16.npz \
    --op_info squeezenet_v1.1_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vv

# VERDICT
echo $0 PASSED
