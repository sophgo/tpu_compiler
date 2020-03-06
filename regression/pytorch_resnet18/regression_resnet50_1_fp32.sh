#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

# use python to mlir , gen golden too
python resnet50_pytorch_to_mlir.py

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename resnet50_op_info.csv \
      resnet50.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter resnet50.mlir \
      --tensor-in resnet50_in_fp32.npz \
      --tensor-out resnet50_out_fp32.npz \
      --dump-all-tensor=resnet50_tensor_all_fp32.npz
  npz_compare.py resnet50_out_fp32.npz resnet50_out_fp32_prob.npz -v
  npz_compare.py \
      resnet50_tensor_all_fp32.npz \
      resnet50_blobs.npz \
      --op_info resnet50_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi

mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename resnet50_op_info.csv \
    --convert-bn-to-scale \
    resnet50.mlir \
    -o resnet50_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter resnet50_opt.mlir \
    --tensor-in input.npz \
    --tensor-out resnet50_out_fp32.npz \
    --dump-all-tensor=resnet50_tensor_all_fp32.npz

npz_compare.py resnet50_out_fp32.npz golden_output.npz -vvv


# VERDICT
echo $0 PASSED
