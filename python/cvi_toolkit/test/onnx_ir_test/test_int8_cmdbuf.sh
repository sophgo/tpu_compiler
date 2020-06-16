#!/bin/bash

if [ x$1 = x ]; then
    echo "Error: no parameter"
    exit 1
fi

echo "test case: $1"

# test fp32
mlir-opt ${1}.mlir \
    --assign-layer-id \
    --assign-chip-name \
     --chipname cv1880v2 \
    --convert-bn-to-scale \
    --canonicalize \
    --eltwise-early-stride \
    --print-tpu-op-info \
    --tpu-op-info-filename ${1}_op_info.csv \
    -o ${1}_fp32.mlir
if [ $? != 0 ]; then
    exit 1
fi

mlir-tpu-interpreter ${1}_fp32.mlir \
    --tensor-in ${1}_input.npz \
    --tensor-out ${1}_output_fp32.npz \
    --dump-all-tensor=${1}_tensor_all_fp32.npz
if [ $? != 0 ]; then
    exit 1
fi

cvi_npz_tool.py compare \
    ${1}_onnx_all_fp32.npz \
    ${1}_tensor_all_fp32.npz \
    --op_info ${1}_op_info.csv \
    --tolerance=0.999,0.999,0.998 -vv
if [ $? != 0 ]; then
    exit 1
fi

# test int8 interpreter
mlir-opt \
    --import-calibration-table \
    --calibration-table ${1}_cali_table \
    ${1}_fp32.mlir \
    -o ${1}_cali.mlir
if [ $? != 0 ]; then
    exit 1
fi

mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename ${1}_op_info_int8_multiplier.csv \
    ${1}_cali.mlir \
    -o ${1}_quant_int8_multiplier.mlir
if [ $? != 0 ]; then
    exit 1
fi

mlir-tpu-interpreter ${1}_quant_int8_multiplier.mlir \
    --tensor-in ${1}_input.npz \
    --tensor-out ${1}_out_int8_multiplier.npz \
    --dump-all-tensor=${1}_tensor_all_int8_multiplier.npz
if [ $? != 0 ]; then
    exit 1
fi

cvi_npz_tool.py compare \
    ${1}_tensor_all_int8_multiplier.npz \
    ${1}_onnx_all_fp32.npz \
    --op_info ${1}_op_info_int8_multiplier.csv \
    --dequant \
    --tolerance=0.5,0.5,0.5 \
    -vv \
    --stats_int8_tensor
if [ $? != 0 ]; then
    exit 1
fi

# test int8 cmdbuf
mlir-opt \
    --tpu-lower --reorder-op \
    ${1}_quant_int8_multiplier.mlir \
    -o ${1}_quant_int8_multiplier_tg.mlir
if [ $? != 0 ]; then
    exit 1
fi

mlir-opt \
    --tg-fuse-leakyrelu --conv-ic-alignment \
    ${1}_quant_int8_multiplier_tg.mlir \
    -o ${1}_quant_int8_multiplier_tg_opt.mlir
if [ $? != 0 ]; then
    exit 1
fi

mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=${1}_weight_map_int8_multiplier.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=${1}_neuron_map_int8_multiplier.csv \
    ${1}_quant_int8_multiplier_tg_opt.mlir \
    -o ${1}_quant_int8_multiplier_addr.mlir
if [ $? != 0 ]; then
    exit 1
fi

mlir-opt \
    --divide-ops-to-func \
    ${1}_quant_int8_multiplier_addr.mlir \
    -o ${1}_quant_int8_multiplier_addr_func.mlir
if [ $? != 0 ]; then
    exit 1
fi

mlir-translate \
    --mlir-to-cvimodel \
    --cvi-set-chip cv1880v2 \
    --weight-file weight_int8_multiplier.bin \
    ${1}_quant_int8_multiplier_addr_func.mlir \
    -o ${1}_int8_multiplier.cvimodel
if [ $? != 0 ]; then
    exit 1
fi

model_runner \
    --dump-all-tensors \
    --input ${1}_input.npz \
    --model ${1}_int8_multiplier.cvimodel \
    --output ${1}_cmdbuf_out_all_int8_multiplier.npz
if [ $? != 0 ]; then
    exit 1
fi

cvi_npz_tool.py compare \
    ${1}_cmdbuf_out_all_int8_multiplier.npz \
    ${1}_tensor_all_int8_multiplier.npz \
    --op_info ${1}_op_info_int8_multiplier.csv
