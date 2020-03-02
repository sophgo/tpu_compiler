#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py \
    ${NET}_tensor_all_int8_multiplier.npz \
    data \
    ${NET}_in_int8.bin \
    int8

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    ${NET}_quant_int8_per_layer.mlir \
    -o ${NET}_quant_int8_per_layer_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    ${NET}_quant_int8_per_layer_tg.mlir \
    -o ${NET}_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --neuron_map neuron_map.csv \
    --output=${NET}_int8_per_layer.cvimodel

# run cvimodel
test_cvinet \
    ${NET}_in_int8.bin \
    ${NET}_int8_per_layer.cvimodel \
    ${NET}_cmdbuf_out_all_int8_per_layer.bin

bin_to_npz.py \
    ${NET}_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    ${NET}_cmdbuf_out_all_int8_per_layer.npz
npz_to_bin.py \
    ${NET}_cmdbuf_out_all_int8_per_layer.npz \
    ${OUTPUTS} \
    ${NET}_cmdbuf_out_${OUTPUTS}_int8_per_layer.bin \
    int8
bin_compare.py \
    ${NET}_cmdbuf_out_${OUTPUTS}_int8_per_layer.bin \
    ${NET}_out_${OUTPUTS}_int8_per_layer.bin \
    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    ${NET}_cmdbuf_out_all_int8_per_layer.npz \
    ${NET}_tensor_all_int8_per_layer.npz \
    --op_info ${NET}_op_info_int8_per_layer.csv

################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    ${NET}_quant_int8_multiplier.mlir \
    -o ${NET}_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    ${NET}_quant_int8_multiplier_tg.mlir \
    -o ${NET}_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --neuron_map neuron_map.csv \
    --output=${NET}_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    ${NET}_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    ${NET}_cmdbuf_out_all_int8_multiplier.bin \
#    16460784 0 16460784 1
test_cvinet \
    ${NET}_in_int8.bin \
    ${NET}_int8_multiplier.cvimodel \
    ${NET}_cmdbuf_out_all_int8_multiplier.bin

bin_to_npz.py \
    ${NET}_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    ${NET}_cmdbuf_out_all_int8_multiplier.npz
npz_to_bin.py \
    ${NET}_cmdbuf_out_all_int8_multiplier.npz \
    ${OUTPUTS} \
    ${NET}_cmdbuf_out_${OUTPUTS}_int8_multiplier.bin \
    int8
bin_compare.py \
    ${NET}_cmdbuf_out_${OUTPUTS}_int8_multiplier.bin \
    ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    ${NET}_cmdbuf_out_all_int8_multiplier.npz \
    ${NET}_tensor_all_int8_multiplier.npz \
    --op_info ${NET}_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
