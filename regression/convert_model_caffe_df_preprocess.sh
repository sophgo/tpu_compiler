#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: $0 prototxt caffemodel raw_scale mean scale swap_channel batch_size cali_table out.cvimodel"
   echo $#
   exit 1
}

if [[ $# -ne 9 ]]; then
    echo "$0 Illegal number of parameters"
    usage
    exit 2
fi

ONE=1

if [ $ONE -eq 1 ]; then

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
    --tpu-lower | \
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
    --weight-file weight.bin \
    int8_tl_lw_memopt_func.mlir \
    -o $9

else

mlir-translate \
    --caffe-to-mlir $1 \
    --caffemodel $2 \
    --resolve-preprocess \
    --raw_scale $3 \
    --mean $4 \
    --scale $5 \
    --swap_channel $6 \
    --static-batchsize $7 \
    -o fp32.mlir

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
    --calibration-table $8 \
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

mlir-opt \
    --divide-ops-to-func \
    int8_tl_lw.mlir \
    -o int8_tl_lw_func.mlir

mlir-translate \
    --mlir-to-cvimodel \
    --weight-file weight.bin \
    int8_tl_lw_func.mlir \
    -o $9

fi
