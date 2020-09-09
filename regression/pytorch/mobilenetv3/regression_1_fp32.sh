#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1
MODEL=$2

cvi_model_convert.py \
    --model_type onnx \
    --model_path $MODEL \
    --model_name ${NET} \
    --mlir_file_path ${NET}.mlir

# MUST 1st prioity than pass_xxx_scale
mlir-opt \
    --skip-mult-used-scale-op \
    --canonicalize \
    --convert-bn-to-scale \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    ${NET}.mlir \
    -o ${NET}_opt.mlir

# test frontend optimizations
#gdb --args \
mlir-tpu-interpreter ${NET}_opt.mlir \
    --debug \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz

# rename onnx output
cvi_npz_tool.py rename \
    ${NET}_out_fp32.npz \
    output_Gemm \
    output

cvi_npz_tool.py compare \
    ${NET}_out_fp32.npz \
    ${NET}_out_onnx.npz -vvv

#cvi_npz_tool.py compare mobilenetv3_pytorch_tensor_all_fp32.npz mobilenetv3_pytorch_out_onnx.npz.all.npz --tolerance=0.9,0.9,.09 --op_info mobilenetv3_pytorch_op_info.csv

# VERDICT
echo $0 PASSED
