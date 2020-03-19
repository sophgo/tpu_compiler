#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.prototxt \
    --caffemodel $MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.caffemodel \
    -o efficientnet_b0.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename efficientnet_b0_op_info.csv \
      efficientnet_b0.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter efficientnet_b0.mlir \
      --tensor-in efficientnet_in_fp32.npz \
      --tensor-out efficientnet_out_fp32.npz \
      --dump-all-tensor=efficientnet_tensor_all_fp32.npz
  npz_tool.py compare efficientnet_out_fp32.npz efficientnet_out_fp32_prob.npz -v
  npz_tool.py compare \
      efficientnet_tensor_all_fp32.npz \
      efficientnet_blobs.npz \
      --op_info efficientnet_b0_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi

# apply opt
# Notes: convert-bn-to-scale has to be done before canonicalizer
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename efficientnet_b0_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    efficientnet_b0.mlir \
    -o efficientnet_b0_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet_b0_opt.mlir \
    --tensor-in efficientnet_in_fp32.npz  \
    --tensor-out efficientnet_out_fp32.npz \
    --dump-all-tensor=efficientnet_tensor_all_fp32.npz

npz_tool.py compare efficientnet_out_fp32.npz efficientnet_out_fp32_prob.npz -v
npz_tool.py compare \
      efficientnet_tensor_all_fp32.npz \
      efficientnet_blobs.npz \
      --op_info efficientnet_b0_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv

# VERDICT
echo $0 PASSED
