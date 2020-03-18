#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py ${NET}_in_fp32.npz ${INPUT} ${NET}_in_fp32.bin

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
    --tpu-weight-map-filename=${NET}_weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=${NET}_neuron_map_bf16.csv \
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
    --output=${NET}_bf16.cvimodel

# run cvimodel
model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_bf16.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_bf16.npz

# compare all tensors
npz_compare.py \
    ${NET}_cmdbuf_out_all_bf16.npz \
    ${NET}_tensor_all_bf16.npz \
    --op_info ${NET}_op_info_bf16.csv \
    --tolerance=$TOLERANCE_BF16_CMDBUF -vv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
  cp ${NET}_bf16.cvimodel $CVIMODEL_REL_PATH
  cp ${NET}_tensor_all_bf16.npz $CVIMODEL_REL_PATH
  cp ${NET}_neuron_map_bf16.csv $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
