#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

################################
# prepare bf16 input
################################
npz_to_bin.py inception_v4_in_fp32.npz input inception_v4_in_fp32.bin
bin_fp32_to_bf16.py \
    inception_v4_in_fp32.bin \
    inception_v4_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    inception_v4_quant_bf16.mlir \
    -o inception_v4_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    inception_v4_quant_bf16_tg.mlir \
    -o inception_v4_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    inception_v4_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir inception_v4_quant_bf16_addr.mlir \
    --output=inception_v4_bf16.cvimodel

## run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    inception_v4_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    inception_v4_cmdbuf_out_all_bf16.bin \
#    54587952 0 54587952 1
model_runner \
    --dump-all-tensors \
    --input inception_v4_in_bf16.bin \
    --model inception_v4_bf16.cvimodel \
    --output inception_v4_cmdbuf_out_all_bf16.bin

bin_to_npz.py \
    inception_v4_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    inception_v4_cmdbuf_out_all_bf16.npz
npz_to_bin.py \
    inception_v4_cmdbuf_out_all_bf16.npz \
    classifier \
    inception_v4_cmdbuf_out_classifier_bf16.bin \
    bf16
bin_compare.py \
    inception_v4_cmdbuf_out_classifier_bf16.bin \
    $REGRESSION_PATH/inception_v4/data/test_cat_out_inception_v4_classifier_bf16.bin \
    bf16 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    inception_v4_cmdbuf_out_all_bf16.npz \
    inception_v4_tensor_all_bf16.npz \
    --order neuron_map_bf16.csv \
    --tolerance=0.99,0.99,0.90 -vvv

# VERDICT
echo $0 PASSED
