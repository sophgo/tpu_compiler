#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/googlenet/caffe/deploy.prototxt \
    --caffemodel $MODEL_PATH/imagenet/googlenet/caffe/bvlc_googlenet.caffemodel \
    -o googlenet.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename googlenet_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    googlenet.mlir \
    -o googlenet_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter googlenet.mlir \
    --tensor-in googlenet_in_fp32.npz \
    --tensor-out googlenet_out_fp32.npz \
    --dump-all-tensor=googlenet_tensor_all_fp32.npz
cvi_npz_tool.py compare googlenet_out_fp32.npz googlenet_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    googlenet_tensor_all_fp32.npz \
    googlenet_blobs.npz \
    --op_info googlenet_op_info.csv

# VERDICT
echo $0 PASSED
