#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/VGG_ILSVRC_16_layers.caffemodel \
    -o vgg16.mlir

# test mlir interpreter
mlir-tpu-interpreter vgg16.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all_fp32.npz
bin_compare.py out.bin $DATA_PATH/test_cat_out_vgg16_prob_fp32.bin \
    float32 1 1 1 1000 5 5

# do only fuse relu optimization
mlir-opt \
    --fuse-relu \
    vgg16.mlir \
    -o vgg16_opt_fuse_relu.mlir


mlir-tpu-interpreter vgg16_opt_fuse_relu.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt_fuse_relu.bin
bin_compare.py out.bin out_opt_fuse_relu.bin float32 1 1 1 1000 5 5

# VERDICT
echo $0 PASSED
