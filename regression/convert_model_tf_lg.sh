#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 tf_model name batch_size cali_table out.cvimodel"
   exit 1
}

if [[ $# -ne 5 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

CUSTOM_OP_PLUGIN_OPTION=""
if [[ ! -z $CUSTOM_OP_PLUGIN ]]; then
    CUSTOM_OP_PLUGIN_OPTION="--custom-op-plugin ${CUSTOM_OP_PLUGIN}"
fi

cvi_model_convert.py \
    --model_path $1 \
    --model_name $2 \
    --model_type tensorflow \
    --batch_size $3 \
    --mlir_file_path fp32.mlir

mlir-opt fp32.mlir \
    --convert-bn-to-scale \
    --canonicalize \
    --eltwise-early-stride \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info.csv \
    -o opt.mlir
mlir-opt \
    opt.mlir \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table $4 | \
mlir-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    ${CUSTOM_OP_PLUGIN_OPTION} \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info_int8.csv \
    -o int8.mlir
mlir-opt \
    int8.mlir \
    --tpu-lower --reorder-op | \
mlir-opt \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment | \
mlir-opt \
    --group-ops | \
mlir-opt \
    --dce \
    --deep-fusion-tg2tl-la \
    --deep-fusion-tl-la2lw \
    -o int8_layergroup.mlir
mlir-opt \
    --compress-weight \
    int8_layergroup.mlir \
    -o int8_layergroup_compressed.mlir
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_int8_lg.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-memory-reuse \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_xxx.csv \
    int8_layergroup_compressed.mlir \
    -o int8_layergroup_addr.mlir
mlir-opt \
    --divide-ops-to-func \
    int8_layergroup_addr.mlir \
    -o int8_layergroup_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    ${CUSTOM_OP_PLUGIN_OPTION} \
    --weight-file weight.bin \
    int8_layergroup_func.mlir \
    -o $5
