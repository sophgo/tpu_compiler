#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel $MODEL_PATH/imagenet/mobilenet_v2/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename mobilenet_v2_op_info.csv \
    mobilenet_v2.mlir \
    -o mobilenet_v2_id.mlir

# test mlir interpreter
mlir-tpu-interpreter mobilenet_v2.mlir \
    --tensor-in mobilenet_v2_in_fp32.npz \
    --tensor-out mobilenet_v2_out_fp32.npz \
    --dump-all-tensor=mobilenet_v2_tensor_all_fp32.npz
cvi_npz_tool.py compare mobilenet_v2_out_fp32.npz mobilenet_v2_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    mobilenet_v2_tensor_all_fp32.npz \
    mobilenet_v2_blobs.npz \
    --op_info mobilenet_v2_op_info.csv \
    --tolerance=0.9999,0.9999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    mobilenet_v2_id.mlir \
    -o mobilenet_v2_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter mobilenet_v2_opt.mlir \
    --tensor-in mobilenet_v2_in_fp32.npz \
    --tensor-out mobilenet_v2_opt_out_fp32.npz
cvi_npz_tool.py compare mobilenet_v2_opt_out_fp32.npz mobilenet_v2_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED
