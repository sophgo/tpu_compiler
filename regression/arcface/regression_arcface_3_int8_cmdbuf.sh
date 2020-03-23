#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

################################
# prepare int8 input
################################
# npz_to_bin.py \
#     arcface_res50_tensor_all_int8_multiplier.npz \
#     data \
#     bmface_in_int8.bin \
#     int8

# don't use following commands to generate input, as it depends on
# calibration result.
#npz_to_bin.py bmface_v3_in_fp32.npz data bmface_in_fp32.bin
#bin_fp32_to_int8.py \
#    bmface_in_fp32.bin \
#    bmface_in_int8.bin \
#    1.0 \
#    0.996580362

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    arcface_res50_quant_int8_multiplier.mlir \
    -o  arcface_res50_quant_int8_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    arcface_res50_quant_int8_tg.mlir \
    -o arcface_res50_quant_int8_cmdbuf.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    arcface_res50_quant_int8_cmdbuf.mlir \
    -o cmdbuf.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir arcface_res50_quant_int8_cmdbuf.mlir \
    --output arcface_res50_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input arcface_res50_in_fp32.npz  \
    --model arcface_res50_int8_multiplier.cvimodel \
    --output arcface_res50_cmdbuf_out_all_int8_multiplier.npz

# compare all tensors

cvi_npz_tool.py compare \
    arcface_res50_cmdbuf_out_all_int8_multiplier.npz \
    arcface_res50_tensor_all_int8_multiplier.npz \
    --op_info arcface_res50_op_info.csv \
    --tolerance 0.9,0.9,0.6 -v

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  NET=arcface_res50
  cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
  mv ${NET}_int8_multiplier.cvimodel $CVIMODEL_REL_PATH
  cp ${NET}_cmdbuf_out_all_int8_multiplier.npz $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED

