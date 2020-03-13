#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################

npz_to_bin.py \
    squeezenet_v1.1_tensor_all_int8_multiplier.npz \
    data_quant \
    squeezenet_v1.1_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# npz_to_bin.py squeezenet_v1.1_in_fp32.npz input squeezenet_v1.1_in_fp32.bin
# bin_fp32_to_int8.py \
#    squeezenet_v1.1_in_fp32.bin \
#    squeezenet_v1.1_in_int8.bin \
#    1.0 \
#    161.008057

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    squeezenet_v1.1_quant_int8_per_layer.mlir \
    -o squeezenet_v1.1_quant_int8_per_layer_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    squeezenet_v1.1_quant_int8_per_layer_tg.mlir \
    -o squeezenet_v1.1_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    squeezenet_v1.1_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --mlir squeezenet_v1.1_quant_int8_per_layer_addr.mlir \
    --output=squeezenet_v1.1_int8_per_layer.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input squeezenet_v1.1_in_fp32.npz \
    --model squeezenet_v1.1_int8_per_layer.cvimodel \
    --output squeezenet_v1.1_cmdbuf_out_all_int8_per_layer.npz

# compare all tensors
npz_compare.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_per_layer.npz \
    squeezenet_v1.1_tensor_all_int8_per_layer.npz \
    --op_info squeezenet_v1.1_op_info_int8_per_layer.csv

################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    squeezenet_v1.1_quant_int8_multiplier.mlir \
    -o squeezenet_v1.1_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    squeezenet_v1.1_quant_int8_multiplier_tg.mlir \
    -o squeezenet_v1.1_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    squeezenet_v1.1_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir squeezenet_v1.1_quant_int8_multiplier_addr.mlir \
    --output=squeezenet_v1.1_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    squeezenet_v1.1_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.bin \
#    16460784 0 16460784 1
model_runner \
    --dump-all-tensors \
    --input squeezenet_v1.1_in_fp32.npz \
    --model squeezenet_v1.1_int8_multiplier.cvimodel \
    --output squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
npz_compare.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.npz \
    squeezenet_v1.1_tensor_all_int8_multiplier.npz \
    --op_info squeezenet_v1.1_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
