#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.prototxt \
    --caffemodel $MODEL_PATH/imagenet/efficientnet-b0/caffe/efficientnet-b0.caffemodel \
    -o efficientnet-b0.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename efficientnet-b0_op_info.csv \
      efficientnet-b0.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter efficientnet-b0.mlir \
      --tensor-in efficientnet-b0_in_fp32.npz \
      --tensor-out efficientnet-b0_out_fp32.npz \
      --dump-all-tensor=efficientnet-b0_tensor_all_fp32.npz
  npz_compare.py efficientnet-b0_out_fp32.npz efficientnet-b0_out_fp32_prob.npz -v
  npz_compare.py \
      efficientnet-b0_tensor_all_fp32.npz \
      efficientnet-b0_blobs.npz \
      --op_info efficientnet-b0_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi

# apply opt
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename efficientnet-b0_op_info.csv \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    --fuse-relu \
    efficientnet-b0.mlir \
    -o efficientnet-b0_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet-b0_opt.mlir \
    -debug-only=interpreter -debug \
    --tensor-in $REGRESSION_PATH/efficientnet-b0/data/efficientnet_in_fp32.npz  \
    --tensor-out efficientnet_out_fp32.npz \
    --dump-all-tensor=efficientnet_tensor_all_fp32.npz
# VERDICT
echo $0 PASSED
