#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

mlir-opt \
    --group-ops \
    ${NET}_quant_int8_multiplier_tg_opt.mlir \
    --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
    -o ${NET}_quant_int8_multiplier_layergroup.mlir

mlir-opt \
    --dce \
    --deep-fusion-tg2tl-la \
    --deep-fusion-tl-la2lw \
    ${NET}_quant_int8_multiplier_layergroup.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_lw.mlir

mlir-opt \
    --compress-weight \
    ${NET}_quant_int8_multiplier_layergroup_lw.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_lw_compressed.mlir

mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=${NET}_weight_map_int8_multiplier_layergroup.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier_layergroup.bin \
    --tpu-generate-compressed-weight \
    ${NET}_quant_int8_multiplier_layergroup_lw_compressed.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_lw_addr.mlir

mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_lg.csv \
    ${NET}_quant_int8_multiplier_layergroup_lw_addr.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_lw_addr_1.mlir

mlir-opt \
    --divide-ops-to-func \
    ${NET}_quant_int8_multiplier_layergroup_lw_addr_1.mlir \
    -o ${NET}_quant_int8_multiplier_layergroup_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    --weight-file weight_int8_multiplier_layergroup.bin \
    ${NET}_quant_int8_multiplier_layergroup_func.mlir \
    -o ${NET}_lg.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_lg.cvimodel \
    --batch-num $BATCH_SIZE \
    --output ${NET}_cmdbuf_out_all_int8_multiplier_lg.npz

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8_multiplier_lg.npz \
    ${NET}_tensor_all_int8_multiplier.npz \
    --op_info ${NET}_op_info_int8_multiplier.csv

echo $0 PASSED
