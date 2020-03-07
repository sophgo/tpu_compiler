#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py ssd300_in_fp32.npz data ssd300_in_fp32.bin
bin_fp32_to_bf16.py \
    ssd300_in_fp32.bin \
    ssd300_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    ssd300_quant_bf16.mlir \
    -o ssd300_quant_bf16_tg.mlir

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
    ssd300_quant_bf16_tg.mlir \
    -o ssd300_quant_bf16_tg_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    ssd300_quant_bf16_tg_addr.mlir \
    -o cmdbuf_bf16.bin


#generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir ssd300_quant_bf16_tg_addr.mlir \
    --output=ssd300_bf16.cvimodel

model_runner \
    --dump-all-tensors \
    --input ssd300_in_bf16.bin \
    --model ssd300_bf16.cvimodel \
    --output ssd300_cmdbuf_out_all_bf16.bin


# # run cmdbuf
# $RUNTIME_PATH/bin/test_bmnet \
#     ssd300_in_bf16.bin \
#     weight_bf16.bin \
#     cmdbuf_bf16.bin \
#     ssd300_cmdbuf_out_all_bf16.bin \
#     32921552 0 32921552 1


# compare all tensors
bin_to_npz.py \
    ssd300_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    ssd300_cmdbuf_out_all_bf16.npz
npz_compare.py \
    ssd300_cmdbuf_out_all_bf16.npz \
    ssd300_tensor_all_bf16.npz \
    --op_info ssd300_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vv

# VERDICT
echo $0 PASSED
