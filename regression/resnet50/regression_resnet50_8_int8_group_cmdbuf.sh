#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# compare all only support when global memory optimization close
COMPARE_ALL=0
################################
# prepare int8 input
################################
cvi_npz_tool.py to_bin \
    resnet50_tensor_all_int8_multiplier.npz \
    data \
    resnet50_in_int8.bin \
    int8
################################
# Lower for quantization 3: multiplier int8
################################
if [ $COMPARE_ALL -eq 1 ]; then
    mlir-opt \
        --group-ops \
        --layer-group-gm-opt=false \
        resnet50_quant_int8_multiplier.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        -o resnet50_quant_int8_multiplier_layergroup.mlir
else
    mlir-opt \
        --group-ops \
        resnet50_quant_int8_multiplier.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        -o resnet50_quant_int8_multiplier_layergroup.mlir
fi

# # # assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_layergroup.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier_layergroup.bin \
    resnet50_quant_int8_multiplier_layergroup.mlir \
    -o resnet50_quant_int8_multiplier_layergroup_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    resnet50_quant_int8_multiplier_layergroup_addr.mlir \
    -o cmdbuf_int8_multiplier_layergroup.bin

# # generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier_layergroup.bin \
    --weight weight_int8_multiplier_layergroup.bin \
    --mlir resnet50_quant_int8_multiplier_layergroup_addr.mlir \
    --output=resnet50_int8_multiplier_layergroup.cvimodel

# # run cmdbuf
# #$RUNTIME_PATH/bin/test_bmnet \
# #    resnet50_in_int8.bin \
# #    weight_int8_multiplier.bin \
# #    cmdbuf_int8_multiplier.bin \
# #    resnet50_cmdbuf_out_all_int8_multiplier_layergroup.bin \
# #    16460784 0 16460784 1


if [ $COMPARE_ALL -eq 1 ]; then
    echo "compare all"
    model_runner \
    --dump-all-tensors \
    --input resnet50_in_fp32.npz \
    --model resnet50_int8_multiplier_layergroup.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_multiplier_layergroup.npz

    # # compare all tensors
    cvi_npz_tool.py compare \
        resnet50_cmdbuf_out_all_int8_multiplier_layergroup.npz \
        resnet50_tensor_all_int8_multiplier.npz \
        --op_info resnet50_op_info_int8_multiplier.csv
else
    model_runner \
    --input resnet50_in_fp32.npz \
    --model resnet50_int8_multiplier_layergroup.cvimodel \
    --output resnet50_cmdbuf_out_all_int8_multiplier_layergroup_fc1000_dequant.npz

    # prepare ref data
    cvi_npz_tool.py extract \
        resnet50_tensor_all_int8_multiplier.npz \
        resnet50_ref_out_fc1000_int8_multiplier.npz \
        fc1000_dequant

    cvi_npz_tool.py compare \
        resnet50_ref_out_fc1000_int8_multiplier.npz \
        resnet50_cmdbuf_out_all_int8_multiplier_layergroup_fc1000_dequant.npz \
        --op_info resnet50_op_info_int8_multiplier.csv
fi

# VERDICT
echo $0 PASSED
