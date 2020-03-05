#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
    --caffemodel $MODEL_PATH/imagenet/vgg/caffe/VGG_ILSVRC_16_layers.caffemodel \
    -o vgg16.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename vgg16_op_info.csv \
    vgg16.mlir \
    -o vgg16_id.mlir

# test mlir interpreter
mlir-tpu-interpreter vgg16.mlir \
    --tensor-in vgg16_in_fp32.npz \
    --tensor-out vgg16_out_fp32.npz \
    --dump-all-tensor=vgg16_tensor_all_fp32.npz
npz_compare.py vgg16_out_fp32.npz vgg16_out_fp32_prob.npz -v
npz_compare.py \
    vgg16_tensor_all_fp32.npz \
    vgg16_blobs.npz \
    --op_info vgg16_op_info.csv \
    --tolerance=0.999,0.999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    vgg16_id.mlir \
    -o vgg16_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter vgg16_opt.mlir \
    --tensor-in vgg16_in_fp32.npz \
    --tensor-out vgg16_opt_out_fp32.npz
npz_compare.py vgg16_opt_out_fp32.npz vgg16_out_fp32_prob.npz -v

# VERDICT
echo $0 PASSED
