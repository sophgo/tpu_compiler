#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 prototxt caffemodel name batch_size raw_scale mean scale swap_channel cali_table out.cvimodel"
   exit 1
}

if [[ $# -ne 9 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

cvi_model_convert.py \
    --model_path $1 \
    --model_dat $2 \
    --model_name $3 \
    --model_type caffe \
    --batch_size $4 \
    --raw_scale $5 \
    --mean $6 \
    --scale $7 \
    --swap_channel $8 \
    --mlir_file_path fp32.mlir

mlir-opt fp32.mlir \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --eltwise-early-stride \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info.csv | \
mlir-opt \
    --import-calibration-table \
    --calibration-table $9 | \
mlir-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    ${CUSTOM_OP_PLUGIN_OPTION} \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info_int8.csv | \
mlir-opt \
    --tpu-lower --reorder-op \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment | \
    --group-ops \
    --layer-group-gm-opt=true \
    --layer-group-neuron-map-filename=neuron_map_layergroup.csv | \
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
    int8_layergroup_compressed.mlir \
    -o int8_layergroup_addr.mlir
mlir-opt \
    --divide-ops-to-func \
    int8_layergroup_addr.mlir \
    -o int8_layergroup_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    --cvi-set-chip ${SET_CHIP_NAME} \
    ${CUSTOM_OP_PLUGIN_OPTION} \
    --weight-file weight.bin \
    int8_layergroup_func.mlir \
    -o ${10}
