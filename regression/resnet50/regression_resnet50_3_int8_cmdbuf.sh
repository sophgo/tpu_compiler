#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py resnet50_in_fp32.npz input resnet50_in_fp32.bin
bin_fp32_to_int8.py \
    resnet50_in_fp32.bin \
    resnet50_in_int8.bin \
    1.0 \
    161.008057

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
    resnet50_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    resnet50_in_int8.bin \
    weight_int8_per_layer.bin \
    cmdbuf_int8_per_layer.bin \
    resnet50_cmdbuf_out_all_int8_per_layer.bin \
    16460784 0 16460784 1
bin_extract.py \
    resnet50_cmdbuf_out_all_int8_per_layer.bin \
    resnet50_cmdbuf_out_fc1000_int8_per_layer.bin \
    int8 0x00024c00 1000
bin_compare.py \
    resnet50_cmdbuf_out_fc1000_int8_per_layer.bin \
    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_per_layer.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    resnet50_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    resnet50_cmdbuf_out_all_int8_per_layer.npz
npz_compare.py \
    resnet50_cmdbuf_out_all_int8_per_layer.npz \
    resnet50_tensor_all_int8_per_layer.npz \
    --order neuron_map.csv

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
    resnet50_quant_int8_multiplier.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_multiplier.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    resnet50_in_int8.bin \
    weight_int8_multiplier.bin \
    cmdbuf_int8_multiplier.bin \
    resnet50_cmdbuf_out_all_int8_multiplier.bin \
    16460784 0 16460784 1
bin_extract.py \
    resnet50_cmdbuf_out_all_int8_multiplier.bin \
    resnet50_cmdbuf_out_fc1000_int8_multiplier.bin \
    int8 0x00024c00 1000
bin_compare.py \
    resnet50_cmdbuf_out_fc1000_int8_multiplier.bin \
    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    resnet50_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    resnet50_cmdbuf_out_all_int8_multiplier.npz
npz_compare.py \
    resnet50_cmdbuf_out_all_int8_multiplier.npz \
    resnet50_tensor_all_int8_multiplier.npz \
    --order neuron_map.csv

# VERDICT
echo $0 PASSED
