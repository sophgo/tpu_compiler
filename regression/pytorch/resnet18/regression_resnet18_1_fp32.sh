#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../../envsetup.sh


convert.py \
    --model_path $MODEL_PATH/imagenet/resnet/onnx/resnet18.onnx \
    --model_name resnet18 \
    --model_type onnx \
    --mlir_file_path resnet18.mlir

mlir-opt \
    --assign-layer-id \
    --canonicalize \
    --convert-bn-to-scale \
    --print-tpu-op-info \
    --tpu-op-info-filename resnet18_op_info.csv \
    resnet18.mlir \
    -o resnet18_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter resnet18_opt.mlir \
    --tensor-in resnet18_in_fp32.npz \
    --tensor-out resnet18_out_fp32.npz \
    --dump-all-tensor=resnet18_tensor_all_fp32.npz

npz_rename.py resnet18_out_fp32.npz output_Gemm output
npz_compare.py resnet18_out_fp32.npz resnet18_out_onnx.npz -vvv


# VERDICT
echo $0 PASSED
