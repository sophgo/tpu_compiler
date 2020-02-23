#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py squeezenet_v1.1_in_fp32.npz data squeezenet_v1.1_in_fp32.bin
bin_fp32_to_int8.py \
    squeezenet_v1.1_in_fp32.bin \
    squeezenet_v1.1_in_int8.bin \
    1.0 \
    151.133789

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
    squeezenet_v1.1_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    squeezenet_v1.1_in_int8.bin \
    weight_int8_per_layer.bin \
    cmdbuf_int8_per_layer.bin \
    squeezenet_v1.1_cmdbuf_out_all_int8_per_layer.bin \
    4838161 0 4838161 1
bin_extract.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_per_layer.bin \
    squeezenet_v1.1_cmdbuf_out_pool10_int8_per_layer.bin \
    int8 0x00025be0 1000
bin_compare.py \
    squeezenet_v1.1_cmdbuf_out_pool10_int8_per_layer.bin \
    $REGRESSION_PATH/squeezenet/data/test_cat_out_squeezenet_v1.1_pool10_int8_per_layer.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    squeezenet_v1.1_cmdbuf_out_all_int8_per_layer.npz
npz_compare.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_per_layer.npz \
    squeezenet_v1.1_tensor_all_int8_per_layer.npz \
    --op_info squeezenet_v1.1_op_info_int8_per_layer.csv

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
    squeezenet_v1.1_quant_int8_multiplier.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_multiplier.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    squeezenet_v1.1_in_int8.bin \
    weight_int8_multiplier.bin \
    cmdbuf_int8_multiplier.bin \
    squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.bin \
    4838161 0 4838161 1
bin_extract.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.bin \
    squeezenet_v1.1_cmdbuf_out_pool10_int8_multiplier.bin \
    int8 0x00025be0 1000
bin_compare.py \
    squeezenet_v1.1_cmdbuf_out_pool10_int8_multiplier.bin \
    $REGRESSION_PATH/squeezenet/data/test_cat_out_squeezenet_v1.1_pool10_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.npz
npz_compare.py \
    squeezenet_v1.1_cmdbuf_out_all_int8_multiplier.npz \
    squeezenet_v1.1_tensor_all_int8_multiplier.npz \
    --op_info squeezenet_v1.1_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
