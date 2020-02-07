#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py ssd300_in_fp32.npz input ssd300_in_fp32.bin
bin_fp32_to_bf16.py \
    ssd300_in_fp32.bin \
    ssd300_in_bf16.bin

################################
# quantization
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    --assign-layer-id \
    ssd300_quant_bf16.mlir \
    -o ssd300_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    ssd300_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin


# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --neuron_map neuron_map_bf16.csv \
    --output=ssd300_bf16.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    resnet50_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    resnet50_cmdbuf_out_all_bf16.bin \
#    32921552 0 32921552 1
$RUNTIME_PATH/bin/test_cvinet \
    ssd300_in_bf16.bin \
    ssd300_bf16.cvimodel \
    ssd300_cmdbuf_out_all_bf16.bin


# # run cmdbuf
# $RUNTIME_PATH/bin/test_bmnet \
#     ssd300_in_bf16.bin \
#     weight_bf16.bin \
#     cmdbuf_bf16.bin \
#     ssd300_cmdbuf_out_all_bf16.bin \
#     32921552 0 32921552 1
# bin_extract.py \
#     ssd300_cmdbuf_out_all_bf16.bin \
#     ssd300_cmdbuf_out_fc1000_bf16.bin \
#     bf16 0x00049800 1000
# bin_compare.py \
#     ssd300_cmdbuf_out_fc1000_bf16.bin \
#     $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_bf16.bin \
#     bf16 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    ssd300_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    ssd300_cmdbuf_out_all_bf16.npz
npz_compare.py \
    ssd300_cmdbuf_out_all_bf16.npz \
    ssd300_tensor_all_bf16.npz \
    --op_info ssd300_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vvv

# VERDICT
echo $0 PASSED
