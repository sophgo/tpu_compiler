#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# create int8 input
npz_tool.py to_bin \
    efficientnet_tensor_all_int8.npz \
    data \
    efficientnet_in_int8.bin \
    int8
# npz_tool.py to_bin efficientnet_in_fp32.npz data efficientnet_in_fp32.bin
# bin_fp32_to_int8.py \
#     efficientnet_in_fp32.bin \
#     efficientnet_in_int8.bin \
#     1.0 \
#     2.64064478874

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
    --output efficientnet_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input efficientnet_in_fp32.npz \
    --model efficientnet_int8_multiplier.cvimodel \
    --output efficientnet_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
npz_tool.py compare \
    efficientnet_tensor_all_int8.npz \
    efficientnet_cmdbuf_out_all_int8_multiplier.npz \
    --op_info efficientnet_b0_op_info.csv

# VERDICT
echo $0 PASSED
