#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# compare all only support when global memory optimization close
COMPARE_ALL=0
################################
# prepare int8 input
################################
npz_tool.py to_bin \
    ${NET}_tensor_all_int8_multiplier.npz \
    data \
    ${NET}_in_int8.bin \
    int8
################################
# Lower for quantization 3: multiplier int8
################################
if [ $COMPARE_ALL -eq 1 ]; then
    mlir-opt \
        --group-ops \
        --layer-group-gm-opt=false \
        ${NET}_quant_int8_multiplier.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        -o ${NET}_quant_int8_multiplier_layergroup.mlir
else
    mlir-opt \
        --group-ops \
        ${NET}_quant_int8_multiplier.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        -o ${NET}_quant_int8_multiplier_layergroup.mlir
fi

# # # assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_layergroup.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier_layergroup.bin \
    ${NET}_quant_int8_multiplier_layergroup.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_int8_multiplier_layergroup_addr.mlir \
    -o cmdbuf_int8_multiplier_layergroup.bin

# # generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier_layergroup.bin \
    --weight weight_int8_multiplier_layergroup.bin \
    --mlir ${NET}_quant_int8_multiplier_layergroup_addr.mlir \
    --output=${NET}_int8_multiplier_layergroup.cvimodel

if [ $COMPARE_ALL -eq 1 ]; then
    echo "compare all"
    model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_multiplier_layergroup.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_layergroup.npz

    # # compare all tensors
    npz_tool.py compare \
        ${NET}_cmdbuf_out_all_int8_multiplier_layergroup.npz \
        ${NET}_tensor_all_int8_multiplier.npz \
        --op_info ${NET}_op_info_int8_multiplier.csv
else
    model_runner \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_multiplier_layergroup.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_layergroup_fc1000_dequant.npz

    # prepare ref data
    npz_tool.py extract \
        ${NET}_tensor_all_int8_multiplier.npz \
        ${NET}_ref_out_fc1000_int8_multiplier.npz \
        fc1000_dequant

    npz_tool.py compare \
        ${NET}_ref_out_fc1000_int8_multiplier.npz \
        ${NET}_cmdbuf_out_all_int8_multiplier_layergroup_fc1000_dequant.npz \
        --op_info ${NET}_op_info_int8_multiplier.csv
fi

# VERDICT
echo $0 PASSED
