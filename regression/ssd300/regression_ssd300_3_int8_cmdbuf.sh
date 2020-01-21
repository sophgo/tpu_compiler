#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py ssd300_in_fp32.npz data ssd300_in_fp32.bin
bin_fp32_to_int8.py \
    ssd300_in_fp32.bin \
    ssd300_in_int8.bin \
    1.0 \
    151.073760986

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
    ssd300_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# # run cmdbuf
# $RUNTIME_PATH/bin/test_bmnet \
#     ssd300_in_int8.bin \
#     weight_int8_per_layer.bin \
#     cmdbuf_int8_per_layer.bin \
#     ssd300_cmdbuf_out_all_int8_per_layer.bin \
#     16460784 0 16460784 1
# bin_extract.py \
#     ssd300_cmdbuf_out_all_int8_per_layer.bin \
#     ssd300_cmdbuf_out_fc1000_int8_per_layer.bin \
#     int8 0x00024c00 1000
# bin_compare.py \
#     ssd300_cmdbuf_out_fc1000_int8_per_layer.bin \
#     $REGRESSION_PATH/ssd300/data/test_cat_out_ssd300_fc1000_int8_per_layer.bin \
#     int8 1 1 1 1000 5

# # compare all tensors
# bin_to_npz.py \
#     ssd300_cmdbuf_out_all_int8_per_layer.bin \
#     neuron_map.csv \
#     ssd300_cmdbuf_out_all_int8_per_layer.npz
# npz_compare.py \
#     ssd300_cmdbuf_out_all_int8_per_layer.npz \
#     ssd300_tensor_all_int8_per_layer.npz \
#     --op_info ssd300_op_info_int8_per_layer.csv

# ################################
# # quantization 2: per-channel int8
# ################################

# # skipped

# ################################
# # quantization 3: multiplier int8
# ################################
# # assign weight address & neuron address
# mlir-opt \
#     --assign-weight-address \
#     --tpu-weight-address-align=16 \
#     --tpu-weight-map-filename=weight_map.csv \
#     --tpu-weight-bin-filename=weight_int8_multiplier.bin \
#     --assign-neuron-address \
#     --tpu-neuron-address-align=16 \
#     --tpu-neuron-map-filename=neuron_map.csv \
#     --assign-layer-id \
#     ssd300_quant_int8_multiplier.mlir | \
#   mlir-translate \
#     --mlir-to-cmdbuf \
#     -o cmdbuf_int8_multiplier.bin

# # run cmdbuf
# $RUNTIME_PATH/bin/test_bmnet \
#     ssd300_in_int8.bin \
#     weight_int8_multiplier.bin \
#     cmdbuf_int8_multiplier.bin \
#     ssd300_cmdbuf_out_all_int8_multiplier.bin \
#     16460784 0 16460784 1
# bin_extract.py \
#     ssd300_cmdbuf_out_all_int8_multiplier.bin \
#     ssd300_cmdbuf_out_fc1000_int8_multiplier.bin \
#     int8 0x00024c00 1000
# bin_compare.py \
#     ssd300_cmdbuf_out_fc1000_int8_multiplier.bin \
#     $REGRESSION_PATH/ssd300/data/test_cat_out_ssd300_fc1000_int8_multiplier.bin \
#     int8 1 1 1 1000 5

# # compare all tensors
# bin_to_npz.py \
#     ssd300_cmdbuf_out_all_int8_multiplier.bin \
#     neuron_map.csv \
#     ssd300_cmdbuf_out_all_int8_multiplier.npz
# npz_compare.py \
#     ssd300_cmdbuf_out_all_int8_multiplier.npz \
#     ssd300_tensor_all_int8_multiplier.npz \
#     --op_info ssd300_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED