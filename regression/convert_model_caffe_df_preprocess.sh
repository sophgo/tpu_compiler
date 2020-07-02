#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 prototxt caffemodel name batch_size raw_scale mean scale swap_channel cali_table out.cvimodel"
   echo $#
   exit 1
}

if [[ $# -ne 10 ]]; then
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
    --convert-func-to-memref | \
mlir-opt \
    --convert-tg-op-to-memref | \
mlir-opt \
    --enable-reuse-global-memory=true \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memopt.csv | \
mlir-opt \
    --convert-tg-op-to-tensor | \
mlir-opt \
    --convert-func-to-tensor \
    -o int8_tl_lw_memopt.mlir
mlir-opt \
    --divide-ops-to-func \
    int8_tl_lw_memopt.mlir \
    -o int8_tl_lw_memopt_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    ${CUSTOM_OP_PLUGIN_OPTION} \
    --weight-file weight.bin \
    int8_tl_lw_memopt_func.mlir \
    -o ${10}
