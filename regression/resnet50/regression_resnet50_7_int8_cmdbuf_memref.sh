#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################

npz_to_bin.py \
    resnet50_tensor_all_int8_multiplier.npz \
    data_quant \
    resnet50_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# npz_to_bin.py resnet50_in_fp32.npz input resnet50_in_fp32.bin
# bin_fp32_to_int8.py \
#    resnet50_in_fp32.bin \
#    resnet50_in_int8.bin \
#    1.0 \
#    161.008057


################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    resnet50_quant_int8_multiplier.mlir \
    -o resnet50_quant_int8_multiplier_tg.mlir

# function argument lower to MemRefType
mlir-opt \
    --debug \
    --convert-func-to-memref \
    resnet50_quant_int8_multiplier_tg.mlir \
    -o resnet50_quant_int8_multiplier_tg_memref.mlir

# op lower to MemRefType
mlir-opt \
    --debug \
    --convert-tg-op-to-memref \
    resnet50_quant_int8_multiplier_tg_memref.mlir \
    -o resnet50_quant_int8_multiplier_tg_op_memref.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    resnet50_quant_int8_multiplier_tg_memref.mlir \
    -o resnet50_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir resnet50_quant_int8_multiplier_addr.mlir \
    --cpufunc_dir ${RUNTIME_PATH}/lib/cpu \
    --output=resnet50_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    resnet50_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    resnet50_cmdbuf_out_all_int8_multiplier.bin \
#    16460784 0 16460784 1
model_runner \
    --model resnet50_in_int8.bin \
    --input resnet50_int8_multiplier.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_multiplier.bin

bin_to_npz.py \
    resnet50_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    resnet50_cmdbuf_out_all_int8_multiplier.npz
npz_to_bin.py \
    resnet50_cmdbuf_out_all_int8_multiplier.npz \
    fc1000 \
    resnet50_cmdbuf_out_fc1000_int8_multiplier.bin \
    int8
bin_compare.py \
    resnet50_cmdbuf_out_fc1000_int8_multiplier.bin \
    $REGRESSION_PATH/resnet50/data/test_cat_out_resnet50_fc1000_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    resnet50_cmdbuf_out_all_int8_multiplier.npz \
    resnet50_tensor_all_int8_multiplier.npz \
    --op_info resnet50_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED


