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
    resnet50_quant_bf16.mlir \
    -o resnet50_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    resnet50_in_bf16.bin \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    resnet50_cmdbuf_out_all_bf16.bin \
    32921552 0 32921552 1
bin_extract.py \
    resnet50_cmdbuf_out_all_bf16.bin \
    resnet50_cmdbuf_out_fc1000_bf16.bin \
    bf16 0x00049800 1000
bin_compare.py \
    resnet50_cmdbuf_out_fc1000_bf16.bin \
    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_bf16.bin \
    bf16 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    resnet50_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    resnet50_cmdbuf_out_all_bf16.npz
npz_compare.py \
    resnet50_cmdbuf_out_all_bf16.npz \
    resnet50_tensor_all_bf16.npz \
    --op_info resnet50_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vvv

# VERDICT
echo $0 PASSED
