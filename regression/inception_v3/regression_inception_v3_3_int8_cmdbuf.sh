#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo $0 IS RUNNING

################################
# prepare int8 input
################################

#cvi_npz_tool.py to_bin \
#    inception_v3_tensor_all_int8_multiplier.npz \
#    data \
#    inception_v3_in_int8.bin \
#    int8


################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    inception_v3_quant_int8_per_layer.mlir \
    -o inception_v3_quant_int8_per_layer_tg.mlir

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
    --convert-cpu-op \
    inception_v3_quant_int8_per_layer_tg.mlir \
    -o inception_v3_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    inception_v3_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
build_cvimodel.py \
    --plugin_dir $INSTALL_PATH/lib/custom_op \
    --plugin_name CustomRuntimeFunc \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --mlir inception_v3_quant_int8_per_layer_addr.mlir \
    --output=inception_v3_int8_per_layer.cvimodel \
    --verbose=1

## run cmdbuf
model_runner \
    --dump-all-tensors \
    --model inception_v3_int8_per_layer.cvimodel \
    --input inception_v3_in_raw_fp32.npz \
    --output inception_v3_cmdbuf_out_all_int8_per_layer.npz

# compare all tensors
cvi_npz_tool.py compare \
    inception_v3_cmdbuf_out_all_int8_per_layer.npz \
    inception_v3_tensor_all_int8_per_layer.npz \
    --op_info inception_v3_op_info_int8_per_layer.csv

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    inception_v3_quant_int8_multiplier.mlir \
    -o inception_v3_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --convert-cpu-op \
    inception_v3_quant_int8_multiplier_tg.mlir \
    -o inception_v3_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    inception_v3_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --plugin_dir $INSTALL_PATH/lib/custom_op \
    --plugin_name CustomRuntimeFunc \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir inception_v3_quant_int8_multiplier_addr.mlir \
    --output=inception_v3_int8_multiplier.cvimodel \
    --verbose=1

## run cmdbuf
model_runner \
    --dump-all-tensors \
    --model inception_v3_int8_multiplier.cvimodel \
    --input inception_v3_in_raw_fp32.npz \
    --output inception_v3_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
cvi_npz_tool.py compare \
    inception_v3_cmdbuf_out_all_int8_multiplier.npz \
    inception_v3_tensor_all_int8_multiplier.npz \
    --op_info inception_v3_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
