#!/bin/bash
set -e

################################
# prepare int8 input
################################
npz_tool.py to_bin \
    test_tensor_all_int8_multiplier.npz \
    data \
    test_in_int8.bin \
    int8

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    test_quant_int8_per_layer.mlir \
    -o test_quant_int8_per_layer_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    test_quant_int8_per_layer_tg.mlir \
    -o test_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    test_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

weight=weight_int8_per_layer.bin
# fake means \build_cvimodel.py MUST assign weight
if [ "$1" = "fake_weight" ]; then
    weight=cmdbuf_int8_per_layer.bin
fi

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight $weight \
    --mlir test_quant_int8_per_layer_addr.mlir \
    --output=test_int8_per_layer.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input test_in_fp32.npz \
    --model test_int8_per_layer.cvimodel \
    --output test_cmdbuf_out_all_int8_per_layer.npz

# compare all tensors
npz_tool.py compare \
    test_cmdbuf_out_all_int8_per_layer.npz \
    test_tensor_all_int8_per_layer.npz \
    --op_info test_op_info_int8_per_layer.csv

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    test_quant_int8_multiplier.mlir \
    -o test_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    test_quant_int8_multiplier_tg.mlir \
    -o test_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    test_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

weight=weight_int8_multiplier.bin
# fake means \build_cvimodel.py MUST assign weight
if [ "$1" = "fake_weight" ]; then
    weight=cmdbuf_int8_multiplier.bin
fi

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight $weight \
    --mlir test_quant_int8_multiplier_addr.mlir \
    --output=test_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input test_in_fp32.npz \
    --model test_int8_multiplier.cvimodel \
    --output test_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
npz_tool.py compare \
    test_cmdbuf_out_all_int8_multiplier.npz \
    test_tensor_all_int8_multiplier.npz \
    --op_info test_op_info_int8_multiplier.csv
