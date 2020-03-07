#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py shufflenet_in_fp32.npz data shufflenet_in_fp32.bin
bin_fp32_to_bf16.py \
    shufflenet_in_fp32.bin \
    shufflenet_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    shufflenet_quant_bf16.mlir \
    -o shufflenet_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    shufflenet_quant_bf16_tg.mlir \
    -o shufflenet_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    shufflenet_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir shufflenet_quant_bf16_addr.mlir \
    --output=shufflenet_bf16.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    shufflenet_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    shufflenet_cmdbuf_out_all_bf16.bin \
#    32921552 0 32921552 1
model_runner \
    --dump-all-tensors \
    --input shufflenet_in_bf16.bin \
    --model shufflenet_bf16.cvimodel \
    --output shufflenet_cmdbuf_out_all_bf16.bin

bin_to_npz.py \
    shufflenet_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    shufflenet_cmdbuf_out_all_bf16.npz

# compare all tensors
npz_compare.py \
    shufflenet_cmdbuf_out_all_bf16.npz \
    shufflenet_tensor_all_bf16.npz \
    --op_info shufflenet_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vv

# VERDICT
echo $0 PASSED
