#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh


################################
# prepare int8 input
################################
npz_to_bin.py densenet_in_fp32.npz input densenet_in_fp32.bin
bin_fp32_to_int8.py \
    densenet_in_fp32.bin \
    densenet_in_int8.bin \
    1.0 \
    2.56927490234

################################
# quantization 1: per-layer int8
################################
# assign weight address & neuron address

# skipped

################################
# quantization 2: per-channel int8
################################

# skipped

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
    densenet_quant_int8_multiplier.mlir \
    -o densenet_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    densenet_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir densenet_quant_int8_multiplier_addr.mlir \
    --cpufunc_dir ${RUNTIME_PATH}/lib/cpu \
    --output=densenet_int8_multiplier.cvimodel

# run cmdbuf
model_runner \
    --input densenet_in_int8.bin \
    --model densenet_int8_multiplier.cvimodel \
    --output densenet_cmdbuf_out_all_int8_multiplier.bin

bin_extract.py \
    densenet_cmdbuf_out_all_int8_multiplier.bin \
    densenet_cmdbuf_out_fc6_int8_multiplier.bin \
    int8 0x00024c00 1000
bin_compare.py \
    densenet_cmdbuf_out_fc6_int8_multiplier.bin \
    densenet_out_fc6_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    densenet_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    densenet_cmdbuf_out_all_int8_multiplier.npz
npz_compare.py \
    densenet_cmdbuf_out_all_int8_multiplier.npz \
    densenet_tensor_all_int8_multiplier.npz \
    --op_info densenet_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
