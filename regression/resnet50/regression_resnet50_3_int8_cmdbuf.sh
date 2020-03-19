#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
cvi_npz_tool.py to_bin \
    resnet50_tensor_all_int8_multiplier.npz \
    data \
    resnet50_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# cvi_npz_tool.py to_bin resnet50_in_fp32.npz input resnet50_in_fp32.bin
# bin_fp32_to_int8.py \
#    resnet50_in_fp32.bin \
#    resnet50_in_int8.bin \
#    1.0 \
#    161.008057

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    resnet50_quant_int8_per_layer.mlir \
    -o resnet50_quant_int8_per_layer_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet50_quant_int8_per_layer_tg.mlir \
    -o resnet50_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --mlir resnet50_quant_int8_per_layer_addr.mlir \
    --output=resnet50_int8_per_layer.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet50_in_fp32.npz \
    --model resnet50_int8_per_layer.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_per_layer.npz

# compare all tensors
cvi_npz_tool.py compare \
    resnet50_cmdbuf_out_all_int8_per_layer.npz \
    resnet50_tensor_all_int8_per_layer.npz \
    --op_info resnet50_op_info_int8_per_layer.csv

################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    resnet50_quant_int8_multiplier.mlir \
    -o resnet50_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet50_quant_int8_multiplier_tg.mlir \
    -o resnet50_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir resnet50_quant_int8_multiplier_addr.mlir \
    --output=resnet50_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input resnet50_in_fp32.npz \
    --model resnet50_int8_multiplier.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    resnet50_cmdbuf_out_all_int8_multiplier.npz \
    resnet50_tensor_all_int8_multiplier.npz \
    --op_info resnet50_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
