#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh


################################
# prepare int8 input
################################
npz_tool.py to_bin ssd300_face_in_fp32.npz data ssd300_face_in_fp32.bin
bin_fp32_to_int8.py \
    ssd300_face_in_fp32.bin \
    ssd300_face_in_int8.bin \
    1.0 \
    177.086471558

mlir-opt \
    --tpu-lower \
    ssd300_face_quant_int8_multiplier.mlir \
    -o ssd300_face_quant_int8_multiplier_tg.mlir

#assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    ssd300_face_quant_int8_multiplier_tg.mlir  \
    -o ssd300_face_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    ssd300_face_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir ssd300_face_quant_int8_multiplier_addr.mlir \
    --cpufunc_dir ${RUNTIME_PATH}/lib/cpu \
    --output=ssd300_face_int8_per_layer.cvimodel

model_runner \
    --input ssd300_face_in_fp32.npz \
    --model ssd300_face_int8_per_layer.cvimodel \
    --output ssd300_face_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors
npz_tool.py compare \
    ssd300_face_cmdbuf_out_all_int8_multiplier.npz \
    ssd300_face_tensor_all_int8_multiplier.npz \
    --op_info ssd300_face_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
