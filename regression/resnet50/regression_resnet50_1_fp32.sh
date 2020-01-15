#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/ResNet-50-deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/ResNet-50-model.caffemodel \
    -o resnet50.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename resnet50_op_info.csv \
    resnet50.mlir \
    -o resnet50_id.mlir

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
    --tolerance=0.9999,0.9999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    resnet50_id.mlir \
    -o resnet50_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter resnet50_opt.mlir \
    --tensor-in resnet50_in_fp32.npz \
    --tensor-out resnet50_opt_out_fp32.npz
npz_compare.py resnet50_opt_out_fp32.npz resnet50_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED
