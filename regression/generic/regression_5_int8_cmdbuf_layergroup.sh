#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CUSTOM_OP_PLUGIN_OPTION=""
if [[ ! -z $CUSTOM_OP_PLUGIN ]]; then
    CUSTOM_OP_PLUGIN_OPTION="--custom-op-plugin ${CUSTOM_OP_PLUGIN}"
fi

# log some data for reference
# resnet50 pass, on chip performance
# Batch 1 97.5 fps
# Batch 2 116 fps
# Batch 4 121.3 fps
# update
# Batch 1 103.6 fps
# Batch 4, 123.4 fps

# mobilenet_v2 pass, on chip performance
# Batch 1, 694.4 fps
# Batch 2, 826.4 fps
# Batch 4, 898.9 fps
# update
# Batch 1, 704.2 fps
# Batch 4, 902.9 fps

# VGG16 pass, on chip performance
# Batch 1, 24.83 fps
# Batch 2, 31.6 fps
# Batch 4, 37.4 fps

# inception_v3 pass, on chip performance
# Batch 1, 82.6 fps
# Batch 4, 95.2 fps

# inception_v4 pass, on chip performance
# Batch 1, 37 fps
# Batch 4, 41.3 fps

# shufflenet_v2 pass, on chip performance
# Batch 1, 1408 fps
# Batch 4, 1619 fps

# googlenet pass, on chip performance
# Batch 1, 201.6 fps
# Batch 4, 228.6 fps

# efficientnet_b0 pass, on chip performance
# Batch 1, 112 fps
# Batch 4, 122 fps
# update
# Batch 1, 310.6 fps
# Batch 4, 405.7 fps
# update
# Batch 1, 346 fps

# yolo_v3_416 pass, on chip performance
# Batch 1, 16 fps
# Batch 4, 16.8 fps
# update
# Batch 1, 16.9fps
# Batch 4, 17.6fps

# resnet18
# Batch 1, 251.7 fps
# Batch 4, 318.7 fps

# sqeezenet
# Batch 1, 892.9 fps
# Batch 4, 1000.0 fps

# compare all only support when global memory optimization close
COMPARE_ALL=0
###############################
# Lower for quantization 3: multiplier int8
###############################
if [ $COMPARE_ALL -eq 1 ]; then
    mlir-opt \
        --convert-cpu-op \
        --group-ops \
        --layer-group-gm-opt=false \
        ${NET}_quant_int8_multiplier_tg_opt.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        --weight-map=weight_map_layergroup.csv \
        --weight-bin=weight_int8_multiplier_layergroup.bin \
        -o ${NET}_quant_int8_multiplier_layergroup.mlir
else
    mlir-opt \
        --convert-cpu-op \
        --group-ops \
        --layer-group-gm-opt=true \
        ${NET}_quant_int8_multiplier_tg_opt.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        --weight-map=weight_map_layergroup.csv \
        --weight-bin=weight_int8_multiplier_layergroup.bin \
        -o ${NET}_quant_int8_multiplier_layergroup.mlir
fi

mlir-opt \
    --dce \
    ${NET}_quant_int8_multiplier_layergroup.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_dce.mlir

mlir-opt \
    --divide-ops-to-func \
    ${NET}_quant_int8_multiplier_layergroup_dce.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_func.mlir

# mlir-opt \
#     --divide-ops-to-func \
#     ${NET}_quant_int8_multiplier_layergroup.mlir \
#     -o ${NET}_quant_int8_multiplier_layergroup_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    ${CUSTOM_OP_PLUGIN_OPTION}\
    --weight-file weight_int8_multiplier_layergroup.bin \
    ${NET}_quant_int8_multiplier_layergroup_func.mlir \
    -o ${NET}_lg.cvimodel

if [ $COMPARE_ALL -eq 1 ]; then
    echo "compare all"
    model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_lg.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_lg.npz

    # # compare all tensors
    cvi_npz_tool.py compare \
        ${NET}_cmdbuf_out_all_int8_multiplier_lg.npz \
        ${NET}_tensor_all_int8_multiplier.npz \
        --op_info ${NET}_op_info_int8_multiplier.csv
else
    echo "compare only output"
    model_runner \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_lg.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_int8_multiplier_lg.npz

    cvi_npz_tool.py compare \
        ${NET}_out_int8_multiplier.npz \
        ${NET}_cmdbuf_out_int8_multiplier_lg.npz \
        --op_info ${NET}_op_info_int8_multiplier.csv
fi

# if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
#   cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_in_fp32.npz
#   cp ${NET}_lg.cvimodel $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_lg.cvimodel
#   cp ${NET}_out_int8_multiplier.npz $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_out_int8_multiplier.npz
# fi

# VERDICT
echo $0 PASSED
