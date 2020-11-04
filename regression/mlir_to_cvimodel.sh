#!/bin/bash
set -xe

if [ $# -lt 2 ]; then
  echo "$1 mlir_file out_cvimodel"
fi

mlir_file=$1
out_cvimodel=$2
output_fp32_results="true"
if [ $# -ge 3 ]; then
  output_fp32_results=$3
fi
optimized_mlir="_lower_opt_$1"
final_mlir="_final_$1"

mlir-opt $mlir_file \
    --tpu-lower \
    --dequant-results-to-fp32=$output_fp32_results \
    --reorder-op \
    --tg-fuse-leakyrelu \
    --conv-ic-alignment \
    --group-ops \
    --dce \
    --deep-fusion-group-slice \
    --deep-fusion-opt \
    -o $optimized_mlir

    #--tg-op-tile \
    #--compress-activation \
mlir-opt $optimized_mlir \
    --compress-weight \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --tpu-generate-compressed-weight \
    --assign-neuron-address \
    --tpu-neuron-memory-reuse \
    --tpu-neuron-address-align=64 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --divide-ops-to-func \
    -o $final_mlir

mlir-translate $final_mlir \
    --mlir-to-cvimodel \
    --weight-file weight.bin \
    -o $out_cvimodel
