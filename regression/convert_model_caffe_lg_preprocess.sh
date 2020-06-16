#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 prototxt caffemodel raw_scale mean scale swap_channel batch_size cali_table out.cvimodel"
   exit 1
}

if [[ $# -ne 9 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

mlir-translate \
    --caffe-to-mlir $1 \
    --caffemodel $2 \
    --resolve-preprocess \
    --raw_scale $3 \
    --mean $4 \
    --scale $5 \
    --swap_channel $6 \
    --static-batchsize $7 | \
mlir-opt \
    --assign-layer-id \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
    --convert-bn-to-scale \
    --canonicalize \
    --eltwise-early-stride \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info.csv | \
mlir-opt \
    --import-calibration-table \
    --calibration-table $8 | \
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info_int8.csv | \
mlir-opt \
    --tpu-lower --reorder-op \
    --group-ops \
    --layer-group-gm-opt=true \
    --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
    --weight-map=weight_map_layergroup.csv \
    --weight-bin=weight.bin \
    -o int8_layergroup.mlir

mlir-opt \
    --divide-ops-to-func \
    int8_layergroup.mlir \
    -o int8_layergroup_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    --cvi-set-chip ${SET_CHIP_NAME} \
    --weight-file weight.bin \
    int8_layergroup_func.mlir \
    -o $9
