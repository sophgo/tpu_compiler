#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 onnxmodel name batch_size cali_table out.cvimodel"
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
    --fuse-relu \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --eltwise-early-stride \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info.csv | \
mlir-opt \
    ${ENABLE_CALI_OVERWRITE_THRESHOLD_FORWARD} \
    --import-calibration-table \
    --calibration-table $4 | \
mlir-opt \
    --assign-chip-name \
    --chipname ${SET_CHIP_NAME} \
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
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv | \
mlir-opt \
    --deep-fusion-tg2tl-la | \
mlir-opt \
    --deep-fusion-tl-la2lw | \
mlir-opt \
    --compress-weight | \
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --tpu-generate-compressed-weight | \
mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-memory-reuse \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv | \
mlir-opt \
    --convert-func-to-tensor \
    -o int8_tl_lw_memopt.mlir
mlir-opt \
    --divide-ops-to-func \
    int8_tl_lw_memopt.mlir \
    -o int8_tl_lw_memopt_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    --weight-file weight.bin \
    int8_tl_lw_memopt_func.mlir \
    -o $5
