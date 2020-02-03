#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

TENSOR_IN_FILE=./data/bmface_v3_in_fp32_scale.npz
TENSOR_IN_BIN_FILE=./data/bmface_v3_in_int8_scale.bin

################################
# prepare int8 input
################################
npz_to_bin.py $TENSOR_IN_FILE data ./data/bmface_v3_in_fp32_scale.bin
bin_fp32_to_int8.py \
    ./data/bmface_v3_in_fp32_scale.bin \
    $TENSOR_IN_BIN_FILE \
    1.0 \
    0.996580362

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
    bmface-v3_quant_int8_multiplier.mlir | \
  mlir-translate \
    -debug \
    -debug-only=mlir-to-cmdbuf,bmnet_bm1880v2_bmkernel_relu \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_multiplier.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    $TENSOR_IN_BIN_FILE \
    weight_int8_multiplier.bin \
    cmdbuf_int8_multiplier.bin \
    bmface-v3_cmdbuf_out_all_int8_multiplier.bin \
    13987584 0 13987584 1

#bin_extract.py \
#    resnet50_cmdbuf_out_all_int8_multiplier.bin \
#    resnet50_cmdbuf_out_fc1000_int8_multiplier.bin \
#    int8 0x00024c00 1000
#bin_compare.py \
#    resnet50_cmdbuf_out_fc1000_int8_multiplier.bin \
#    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_multiplier.bin \
#    int8 1 1 1 1000 5


# compare all tensors
bin_to_npz.py \
    bmface-v3_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    bmface-v3_cmdbuf_out_all_int8_multiplier.npz

npz_compare.py \
    bmface-v3_cmdbuf_out_all_int8_multiplier.npz \
    bmface-v3_tensor_all_int8_multiplier.npz \
    --op_info bmface-v3_op_info_int8_multiplier.csv


# VERDICT
echo $0 PASSED


#################################
## quantization 1: per-layer int8
#################################
## assign weight address & neuron address
#mlir-opt \
#    --assign-weight-address \
#    --tpu-weight-address-align=16 \
#    --tpu-weight-map-filename=weight_map.csv \
#    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
#    --assign-neuron-address \
#    --tpu-neuron-address-align=16 \
#    --tpu-neuron-map-filename=neuron_map.csv \
#    --assign-layer-id \
#    resnet50_quant_int8_per_layer.mlir | \
#  mlir-translate \
#    --mlir-to-cmdbuf \
#    -o cmdbuf_int8_per_layer.bin
#
## run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    resnet50_in_int8.bin \
#    weight_int8_per_layer.bin \
#    cmdbuf_int8_per_layer.bin \
#    resnet50_cmdbuf_out_all_int8_per_layer.bin \
#    16460784 0 16460784 1
#bin_extract.py \
#    resnet50_cmdbuf_out_all_int8_per_layer.bin \
#    resnet50_cmdbuf_out_fc1000_int8_per_layer.bin \
#    int8 0x00024c00 1000
#bin_compare.py \
#    resnet50_cmdbuf_out_fc1000_int8_per_layer.bin \
#    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_per_layer.bin \
#    int8 1 1 1 1000 5
#
## compare all tensors
#bin_to_npz.py \
#    resnet50_cmdbuf_out_all_int8_per_layer.bin \
#    neuron_map.csv \
#    resnet50_cmdbuf_out_all_int8_per_layer.npz
#npz_compare.py \
#    resnet50_cmdbuf_out_all_int8_per_layer.npz \
#    resnet50_tensor_all_int8_per_layer.npz \
#    --op_info resnet50_op_info_int8_per_layer.csv

################################
# quantization 2: per-channel int8
################################

# skipped
