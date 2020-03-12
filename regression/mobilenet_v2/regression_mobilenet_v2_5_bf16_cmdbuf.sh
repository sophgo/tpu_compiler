#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py mobilenet_v2_in_fp32.npz input mobilenet_v2_in_fp32.bin
bin_fp32_to_bf16.py \
    mobilenet_v2_in_fp32.bin \
    mobilenet_v2_in_bf16.bin

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
    mobilenet_v2_quant_bf16.mlir \
    -o mobilenet_v2_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    mobilenet_v2_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir mobilenet_v2_quant_bf16_addr.mlir \
    --output=mobilenet_v2_bf16.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input mobilenet_v2_in_bf16.bin \
    --model mobilenet_v2_bf16.cvimodel \
    --output mobilenet_v2_cmdbuf_out_all_bf16.npz

# compare all tensors
npz_compare.py \
    mobilenet_v2_cmdbuf_out_all_bf16.npz \
    mobilenet_v2_tensor_all_bf16.npz \
    --op_info mobilenet_v2_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vvv

# VERDICT
echo $0 PASSED
