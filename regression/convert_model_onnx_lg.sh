#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 prototxt caffemodel batch_size cali_table out.cvimodel"
   exit 1
}

if [[ $# -ne 5 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

cvi_model_convert.py \
    --model_path $1 \
    --model_name $2 \
    --model_type onnx \
    --batch_size $3 \
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
    --calibration-table $4 | \
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info_int8.csv | \
mlir-opt \
    --convert-cpu-op \
    --group-ops \
    --layer-group-gm-opt=true \
    --layer-group-neuron-map-filename=neuron_map_layergroup.csv \
    --weight-map=weight_map_layergroup.csv \
    --weight-bin=weight.bin \
    -o int8_layergroup.mlir

mlir-translate \
    int8_layergroup.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir int8_layergroup.mlir \
    --output=$5
