#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py vgg16_in_fp32.npz input vgg16_in_fp32.bin
bin_fp32_to_bf16.py \
    vgg16_in_fp32.bin \
    vgg16_in_bf16.bin

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
    vgg16_quant_bf16.mlir \
    -o vgg16_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    vgg16_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir vgg16_quant_bf16_addr.mlir \
    --cpufunc_dir ${RUNTIME_PATH}/lib/cpu \
    --output=vgg16_bf16.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    vgg16_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    vgg16_cmdbuf_out_all_bf16.bin \
#    32921552 0 32921552 1
model_runner \
    --input vgg16_in_bf16.bin \
    --model vgg16_bf16.cvimodel \
    --output vgg16_cmdbuf_out_all_bf16.bin

bin_extract.py \
    vgg16_cmdbuf_out_all_bf16.bin \
    vgg16_cmdbuf_out_fc8_bf16.bin \
    bf16 0x00049800 1000
bin_compare.py \
    vgg16_cmdbuf_out_fc8_bf16.bin \
    $REGRESSION_PATH/vgg16/data/test_cat_out_vgg16_fc8_bf16.bin \
    bf16 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    vgg16_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    vgg16_cmdbuf_out_all_bf16.npz
npz_compare.py \
    vgg16_cmdbuf_out_all_bf16.npz \
    vgg16_tensor_all_bf16.npz \
    --op_info vgg16_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vvv

# VERDICT
echo $0 PASSED
