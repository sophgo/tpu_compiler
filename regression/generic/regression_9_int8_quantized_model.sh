#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

cvi_model_convert.py \
    --model_path $MODEL_DEF \
    --model_dat "$MODEL_DAT" \
    --model_name ${NET} \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --mlir_file_path ${NET}.mlir

tpuc-opt ${NET}.mlir \
    --canonicalize \
    --fuse-pad \
    --assign-chip-name \
    --print-tpu-op-info \
    --tpu-op-info-filename \
    ${NET}_op_info.csv \
    -o ${NET}_quant.mlir

# there are not all tl operation ready, layer group also not ready
# we generic to cvimodel manually
optimized_mlir="_lower_opt_${NET}_quant.mlir"
final_mlir="_final_${NET}_quant.mlir"

tpuc-opt ${NET}_quant.mlir\
    --tpu-lower \
    --dequant-results-to-fp32=1 \
    --reorder-op \
    --conv-ic-alignment \
    --dce \
    --deep-fusion-opt \
    -o $optimized_mlir

# compress activation will be failed, close
tpuc-opt $optimized_mlir \
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

tpuc-translate $final_mlir \
    --mlir-to-cvimodel \
    --weight-file weight.bin \
    -o ${NET}_int8.cvimodel

model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8.npz

tpuc-interpreter ${NET}_quant.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_int8.npz \
    --dump-all-tensor=${NET}_tensor_all_int8.npz

cvi_npz_tool.py compare \
    ${NET}_cmdbuf_out_all_int8.npz \
    ${NET}_tensor_all_int8.npz \
    --op_info ${NET}_op_info.csv -vv


echo $0 PASSED
