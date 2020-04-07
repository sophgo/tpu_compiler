#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo $0 IS RUNNING

# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    inception_v3_opt.mlir \
    -o inception_v3_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter  inception_v3_quant_bf16.mlir \
    --tensor-in inception_v3_in_raw_fp32.npz \
    --tensor-out inception_v3_out_bf16.npz \
    --customer-interpret-plugin ~/work/install_host/lib/cpu/CustomerInterpret.so \
    --dump-all-tensor=inception_v3_tensor_all_bf16.npz
cvi_npz_tool.py compare inception_v3_out_bf16.npz inception_v3_out_fp32.npz -v
# need to check torlerance later
cvi_npz_tool.py compare \
    inception_v3_tensor_all_bf16.npz \
    inception_v3_tensor_all_fp32.npz \
    --op_info inception_v3_op_info.csv \
    --tolerance=0.99,0.99,0.88 -vvv

# VERDICT
echo $0 PASSED
