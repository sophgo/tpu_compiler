#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    --print-tpu-op-info \
    --tpu-op-info-filename yolo_v3_op_info_bf16_per_layer.csv \
    yolo_v3_416_opt.mlir \
    -o yolo_v3_416_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter yolo_v3_416_quant_bf16.mlir \
    --tensor-in yolo_v3_in_fp32.npz \
    --tensor-out yolo_v3_out_bf16.npz \
    --dump-all-tensor=yolo_v3_416_tensor_all_bf16.npz

npz_compare.py \
    yolo_v3_416_tensor_all_bf16.npz \
    yolo_v3_tensor_all_fp32.npz \
    --op_info yolo_v3_op_info.csv \
    --tolerance=0.99,0.99,0.94 -vv

# VERDICT
echo $0 PASSED
