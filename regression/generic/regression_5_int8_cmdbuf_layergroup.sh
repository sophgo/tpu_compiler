#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# log some data for reference
# resnet50 pass, on chip performance
# Batch 1 97.5 fps
# Batch 2 116 fps
# Batch 4 121.3 fps

# mobilenet_v2 pass, on chip performance
# Batch 1, 694.4 fps
# Batch 2, 826.4 fps
# Batch 4, 898.9 fps

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
        ${NET}_quant_int8_multiplier.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        --weight-map=weight_map_layergroup.csv \
        --weight-bin=weight_int8_multiplier_layergroup.bin \
        -o ${NET}_quant_int8_multiplier_layergroup.mlir
else
    mlir-opt \
        --convert-cpu-op \
        --group-ops \
        --layer-group-gm-opt=true \
        ${NET}_quant_int8_multiplier.mlir \
        --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
        --weight-map=weight_map_layergroup.csv \
        --weight-bin=weight_int8_multiplier_layergroup.bin \
        -o ${NET}_quant_int8_multiplier_layergroup.mlir
fi

mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_int8_multiplier_layergroup.mlir \
    -o cmdbuf_int8_multiplier_layergroup.bin

# # generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier_layergroup.bin \
    --weight weight_int8_multiplier_layergroup.bin \
    --mlir ${NET}_quant_int8_multiplier_layergroup.mlir \
    --output=${NET}_lg.cvimodel

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

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  if [ $BATCH_SIZE -eq 1 ]; then
    mv ${NET}_lg.cvimodel $CVIMODEL_REL_PATH
  else
    mv ${NET}_lg.cvimodel $CVIMODEL_REL_PATH/${NET}_bs${BATCH_SIZE}_lg.cvimodel
  fi
fi

# VERDICT
echo $0 PASSED
