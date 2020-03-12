#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py vgg16_in_fp32.npz input vgg16_in_fp32.bin
bin_fp32_to_int8.py \
    vgg16_in_fp32.bin \
    vgg16_in_int8.bin \
    1.0 \
    161.057006836

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
    --assign-layer-id \
    vgg16_quant_int8_per_layer.mlir \
    -o vgg16_quant_int8_per_layer_addr.mlir \

mlir-translate \
    --mlir-to-cmdbuf \
    vgg16_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --mlir vgg16_quant_int8_per_layer.mlir \
    --output=vgg16_int8_per_layer.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input vgg16_in_int8.bin \
    --model vgg16_int8_per_layer.cvimodel \
    --output vgg16_cmdbuf_out_all_int8_per_layer.npz

# compare all tensors
npz_compare.py \
    vgg16_cmdbuf_out_all_int8_per_layer.npz \
    vgg16_tensor_all_int8_per_layer.npz \
    --op_info vgg16_op_info_int8_per_layer.csv

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# quantization 3: multiplier int8
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    vgg16_quant_int8_multiplier.mlir \
    -o vgg16_quant_int8_multiplier_addr.mlir
  mlir-translate \
    --mlir-to-cmdbuf \
    vgg16_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir vgg16_quant_int8_multiplier_addr.mlir \
    --output=vgg16_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input vgg16_in_int8.bin \
    --model vgg16_int8_multiplier.cvimodel \
    --output vgg16_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
npz_compare.py \
    vgg16_cmdbuf_out_all_int8_multiplier.npz \
    vgg16_tensor_all_int8_multiplier.npz \
    --op_info vgg16_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
