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

ONE=1

if [ $ONE -eq 1 ]; then

cvi_model_convert.py \
    --model_path $1 \
    --model_name $2 \
    --model_type onnx \
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
    --tpu-lower | \
mlir-opt \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment | \
mlir-opt \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --convert-cpu-op | \
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
    --convert-cpu-op \
    -o int8_tl_lw_memopt.mlir

mlir-translate \
    int8_tl_lw_memopt.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir int8_tl_lw_memopt.mlir \
    --output=$5

else

cvi_model_convert.py \
    --model_path $1 \
    --model_name $2 \
    --model_type onnx \
    --mlir_file_path fp32.mlir

mlir-opt \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --eltwise-early-stride \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info.csv \
    fp32.mlir \
    -o fp32_opt.mlir

mlir-opt \
    --import-calibration-table \
    --calibration-table $4 \
    fp32_opt.mlir \
    -o cali.mlir

mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename op_info_int8.csv \
    cali.mlir \
    -o int8.mlir

mlir-opt \
    --tpu-lower \
    int8.mlir \
    -o int8_tg.mlir

mlir-opt \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment \
    int8_tg.mlir \
    -o int8_tg_opt.mlir

mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --convert-cpu-op \
    int8_tg_opt.mlir \
    -o int8_addr.mlir

mlir-opt \
    --deep-fusion-tg2tl-la \
    int8_addr.mlir \
    -o int8_tl_la.mlir

mlir-opt \
    --deep-fusion-tl-la2lw \
    int8_tl_la.mlir \
    -o int8_tl_lw.mlir

mlir-translate \
    int8_tl_lw.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf.bin

build_cvimodel.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --mlir int8_tl_lw.mlir \
    --output=$5

fi
