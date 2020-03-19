#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/densenet/caffe/densenet121_deploy.prototxt \
    --caffemodel $MODEL_PATH/imagenet/densenet/caffe/densenet121.caffemodel \
    -o densenet.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename densenet_op_info.csv \
    densenet.mlir \
    -o densenet_id.mlir

# test mlir interpreter
mlir-tpu-interpreter densenet.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_out_fp32.npz \
    --dump-all-tensor=densenet_tensor_all_fp32.npz
npz_tool.py compare densenet_out_fp32.npz densenet_out_fp32_fc6.npz -vvv
npz_tool.py compare \
    densenet_tensor_all_fp32.npz \
    densenet_blobs.npz \
    --op_info densenet_op_info.csv \
    --tolerance=0.9999,0.9999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    densenet_id.mlir \
    -o densenet_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter densenet_opt.mlir \
    --tensor-in densenet_in_fp32.npz \
    --tensor-out densenet_opt_out_fp32.npz
npz_tool.py compare densenet_opt_out_fp32.npz densenet_out_fp32_fc6.npz -v

# VERDICT
echo $0 PASSED
