#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3_deploy.prototxt \
    --caffemodel $MODEL_PATH/imagenet/shufflenet_v1/caffe/shufflenet_1x_g3.caffemodel \
    -o shufflenet.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename shufflenet_op_info.csv \
    shufflenet.mlir \
    -o shufflenet_id.mlir

# test mlir interpreter
mlir-tpu-interpreter shufflenet.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_out_fp32.npz \
    --dump-all-tensor=shufflenet_tensor_all_fp32.npz
npz_compare.py shufflenet_out_fp32.npz shufflenet_out_fp32_prob.npz -v
npz_compare.py \
    shufflenet_tensor_all_fp32.npz \
    shufflenet_blobs.npz \
    --op_info shufflenet_op_info.csv \
    --tolerance=0.9999,0.9999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    shufflenet_id.mlir \
    -o shufflenet_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter shufflenet_opt.mlir \
    --tensor-in shufflenet_in_fp32.npz \
    --tensor-out shufflenet_opt_out_fp32.npz
npz_compare.py shufflenet_opt_out_fp32.npz shufflenet_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED