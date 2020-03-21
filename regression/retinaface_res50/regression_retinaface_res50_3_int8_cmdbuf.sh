#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
# cvi_npz_tool.py to_bin \
#     retinaface_res50_tensor_all_int8.npz \
#     data_quant \
#     retinaface_res50_in_int8.bin \
#     int8

# cvi_npz_tool.py to_bin \
#     retinaface_res50_in_fp32.npz \
#     data \
#     retinaface_res50_in_fp32.bin \
#    float32

#cvi_npz_tool.py to_bin retinaface_res50_in_fp32.npz data retinaface_res50_in_fp32.bin
# Depend on retinaface_res50_threshold_table
#bin_fp32_to_int8.py \
#    retinaface_res50_in_fp32.bin \
#    retinaface_res50_in_int8.bin \
#    1.0 \
#    255.003890991

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    retinaface_res50_quant_int8.mlir \
    -o retinaface_res50_quant_int8_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    retinaface_res50_quant_int8_tg.mlir \
    -o retinaface_res50_quant_int8_addr.mlir

mlir-translate retinaface_res50_quant_int8_addr.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8.bin \
    --weight weight_int8.bin \
    --mlir retinaface_res50_quant_int8_addr.mlir \
    --output=retinaface_res50_int8.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input retinaface_res50_in_fp32.npz \
    --model retinaface_res50_int8.cvimodel \
    --output retinaface_res50_cmdbuf_out_all_int8.npz

# compare all tensors
cvi_npz_tool.py compare \
    retinaface_res50_cmdbuf_out_all_int8.npz \
    retinaface_res50_tensor_all_int8.npz \
    --op_info retinaface_res50_op_info_int8.csv

# VERDICT
echo $0 PASSED
