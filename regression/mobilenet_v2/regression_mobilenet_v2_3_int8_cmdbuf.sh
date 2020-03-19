#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
cvi_npz_tool.py to_bin mobilenet_v2_in_fp32.npz input mobilenet_v2_in_fp32.bin
bin_fp32_to_int8.py \
    mobilenet_v2_in_fp32.bin \
    mobilenet_v2_in_int8.bin \
    1.0 \
    2.56929183

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
    mobilenet_v2_quant_int8_per_layer.mlir \
    --o mobilenet_v2_quant_int8_per_layer_addr.mlir \
  mlir-translate \
    --mlir-to-cmdbuf \
    mobilenet_v2_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --mlir mobilenet_v2_quant_int8_per_layer_addr.mlir \
    --output=mobilenet_v2_int8_per_layer.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input mobilenet_v2_in_fp32.npz \
    --model mobilenet_v2_int8_per_layer.cvimodel \
    --output mobilenet_v2_cmdbuf_out_all_int8_per_layer.npz

# compare all tensors
cvi_npz_tool.py compare \
    mobilenet_v2_cmdbuf_out_all_int8_per_layer.npz \
    mobilenet_v2_tensor_all_int8_per_layer.npz \
    --op_info mobilenet_v2_op_info_int8_per_layer.csv

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
    mobilenet_v2_quant_int8_multiplier.mlir \
    -o mobilenet_v2_quant_int8_multiplier_addr.mlir \
  mlir-translate \
    --mlir-to-cmdbuf \
    mobilenet_v2_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir mobilenet_v2_quant_int8_multiplier_addr.mlir \
    --output=mobilenet_v2_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input mobilenet_v2_in_fp32.npz \
    --model mobilenet_v2_int8_multiplier.cvimodel \
    --output mobilenet_v2_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    mobilenet_v2_cmdbuf_out_all_int8_multiplier.npz \
    mobilenet_v2_tensor_all_int8_multiplier.npz \
    --op_info mobilenet_v2_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
