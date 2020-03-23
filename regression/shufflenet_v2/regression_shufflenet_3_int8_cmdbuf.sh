#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_INT8_PER_LAYER=1

################################
# prepare int8 input
################################

cvi_npz_tool.py to_bin \
    shufflenet_tensor_all_int8_multiplier.npz \
    data_quant \
    shufflenet_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# cvi_npz_tool.py to_bin shufflenet_in_fp32.npz input shufflenet_in_fp32.bin
# bin_fp32_to_int8.py \
#    shufflenet_in_fp32.bin \
#    shufflenet_in_int8.bin \
#    1.0 \
#    161.008057

################################
# Lower for quantization 1: per-layer int8
################################
if [ $COMPARE_INT8_PER_LAYER -eq 1 ]; then
    mlir-opt \
        --tpu-lower \
        shufflenet_quant_int8_per_layer.mlir \
        -o shufflenet_quant_int8_per_layer_tg.mlir

    # assign weight address & neuron address
    mlir-opt \
        --assign-weight-address \
        --tpu-weight-address-align=16 \
        --tpu-weight-map-filename=weight_map.csv \
        --tpu-weight-bin-filename=weight_int8_per_layer.bin \
        --assign-neuron-address \
        --tpu-neuron-address-align=16 \
        --tpu-neuron-map-filename=neuron_map.csv \
        shufflenet_quant_int8_per_layer_tg.mlir \
        -o shufflenet_quant_int8_per_layer_addr.mlir

    mlir-translate \
        --mlir-to-cmdbuf \
        shufflenet_quant_int8_per_layer_addr.mlir \
        -o cmdbuf_int8_per_layer.bin

    # generate cvi model
    build_cvimodel.py \
        --cmdbuf cmdbuf_int8_per_layer.bin \
        --weight weight_int8_per_layer.bin \
        --mlir shufflenet_quant_int8_per_layer_addr.mlir \
        --output=shufflenet_int8_per_layer.cvimodel

    # run cmdbuf
    model_runner \
        --dump-all-tensors \
        --input shufflenet_in_fp32.npz \
        --model shufflenet_int8_per_layer.cvimodel \
        --output shufflenet_cmdbuf_out_all_int8_per_layer.npz

    # compare all tensors
    cvi_npz_tool.py compare \
        shufflenet_cmdbuf_out_all_int8_per_layer.npz \
        shufflenet_tensor_all_int8_per_layer.npz \
        --op_info shufflenet_op_info_int8_per_layer.csv

fi
################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    shufflenet_quant_int8_multiplier.mlir \
    -o shufflenet_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    shufflenet_quant_int8_multiplier_tg.mlir \
    -o shufflenet_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    shufflenet_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir shufflenet_quant_int8_multiplier_addr.mlir \
    --output=shufflenet_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input shufflenet_in_fp32.npz \
    --model shufflenet_int8_multiplier.cvimodel \
    --output shufflenet_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    shufflenet_cmdbuf_out_all_int8_multiplier.npz \
    shufflenet_tensor_all_int8_multiplier.npz \
    --op_info shufflenet_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
