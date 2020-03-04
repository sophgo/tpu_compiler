#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py ${NET}_in_fp32.npz ${INPUT} ${NET}_in_fp32.bin
bin_fp32_to_bf16.py \
    ${NET}_in_fp32.bin \
    ${NET}_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    ${NET}_quant_bf16.mlir \
    -o ${NET}_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    ${NET}_quant_bf16_tg.mlir \
    -o ${NET}_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvimodel
build_cvimodel.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --mlir ${NET}_quant_bf16_addr.mlir \
    --cpufunc_dir ${RUNTIME_PATH}/lib/cpu \
    --output=${NET}_bf16.cvimodel

# run cvimodel
model_runner \
    --input ${NET}_in_bf16.bin \
    --model ${NET}_bf16.cvimodel \
    --output ${NET}_cmdbuf_out_all_bf16.bin

bin_to_npz.py \
    ${NET}_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    ${NET}_cmdbuf_out_all_bf16.npz

# compare all tensors
npz_compare.py \
    ${NET}_cmdbuf_out_all_bf16.npz \
    ${NET}_tensor_all_bf16.npz \
    --op_info ${NET}_op_info_bf16.csv \
    --tolerance=$TOLERANCE_BF16_CMDBUF -vv

# VERDICT
echo $0 PASSED
