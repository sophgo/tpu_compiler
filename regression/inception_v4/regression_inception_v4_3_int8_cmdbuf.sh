#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

################################
# prepare int8 input
################################

npz_tool.py to_bin \
    inception_v4_tensor_all_int8_multiplier.npz \
    data \
    inception_v4_in_int8.bin \
    int8

# npz_tool.py to_bin inception_v4_in_fp32.npz input inception_v4_in_fp32.bin
# bin_fp32_to_int8.py \
#     inception_v4_in_fp32.bin \
#     inception_v4_in_int8.bin \
#     1.0 \
#     0.994933307171

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    inception_v4_quant_int8_per_layer.mlir \
    -o inception_v4_quant_int8_per_layer_tg.mlir

################################
# quantization 1: per-layer int8
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    inception_v4_quant_int8_per_layer_tg.mlir  \
    -o inception_v4_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    inception_v4_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --mlir inception_v4_quant_int8_per_layer_addr.mlir \
    --output=inception_v4_int8_per_layer.cvimodel

## run cmdbuf
model_runner \
    --dump-all-tensors \
    --input inception_v4_in_fp32.npz \
    --model inception_v4_int8_per_layer.cvimodel \
    --output inception_v4_cmdbuf_out_all_int8_per_layer.npz

# compare all tensors
npz_tool.py compare \
    inception_v4_cmdbuf_out_all_int8_per_layer.npz \
    inception_v4_tensor_all_int8_per_layer.npz \
    --op_info inception_v4_op_info_int8_per_layer.csv

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    inception_v4_quant_int8_multiplier.mlir \
    -o inception_v4_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    inception_v4_quant_int8_multiplier_tg.mlir \
    -o inception_v4_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    inception_v4_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir inception_v4_quant_int8_multiplier_addr.mlir \
    --output=inception_v4_int8_multiplier.cvimodel

## run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    inception_v4_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    inception_v4_cmdbuf_out_all_int8_multiplier.bin \
#    27293984 0 27293984 1
model_runner \
    --dump-all-tensors \
    --input inception_v4_in_fp32.npz \
    --model inception_v4_int8_multiplier.cvimodel \
    --output inception_v4_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
npz_tool.py compare \
    inception_v4_cmdbuf_out_all_int8_multiplier.npz \
    inception_v4_tensor_all_int8_multiplier.npz \
    --op_info inception_v4_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
