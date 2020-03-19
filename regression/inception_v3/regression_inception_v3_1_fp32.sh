#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/inception_v3/caffe/deploy_inception-v3.prototxt \
    --caffemodel $MODEL_PATH/imagenet/inception_v3/caffe/inception-v3.caffemodel \
    --add-preprocess \
    --preprocess-transpose \
    --preprocess-mean 128,128,128 \
    --preprocess-scale 0.0078125 \
    --preprocess-color-order 2,1,0 \
    -o inception_v3.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v3_op_info.csv \
    inception_v3.mlir \
    -o dummy.mlir
## test mlir interpreter
mlir-tpu-interpreter inception_v3.mlir \
    --tensor-in inception_v3_in_raw_fp32.npz \
    --tensor-out inception_v3_out_fp32.npz \
    --dump-all-tensor=inception_v3_tensor_all_fp32.npz
cvi_npz_tool.py compare inception_v3_out_fp32.npz inception_v3_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    inception_v3_tensor_all_fp32.npz \
    inception_v3_blobs.npz \
    --op_info inception_v3_op_info.csv \
    --tolerance=0.99,0.99,0.91 -vvv

# assign layer_id right away, and apply all frontend optimizations
# Notes: convert-bn-to-scale has to be done before canonicalizer
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename inception_v3_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    inception_v3.mlir \
    -o inception_v3_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter inception_v3_opt.mlir \
    --tensor-in inception_v3_in_raw_fp32.npz \
    --tensor-out inception_v3_opt_out_fp32.npz \
    --dump-all-tensor=inception_v3_tensor_all_fp32.npz
cvi_npz_tool.py compare inception_v3_opt_out_fp32.npz inception_v3_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    inception_v3_tensor_all_fp32.npz \
    inception_v3_blobs.npz \
    --op_info inception_v3_op_info.csv \
    --tolerance=0.99,0.99,0.99 -vvv

# VERDICT
echo $0 PASSED
