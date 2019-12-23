#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/VGG_ILSVRC_16_layers_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/VGG_ILSVRC_16_layers.caffemodel \
    -o vgg16.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
    --fuse-relu \
    vgg16.mlir \
    -o vgg16_opt.mlir

# fp32 inference
mlir-tpu-interpreter vgg16_opt.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all.npz

# quantization
mlir-opt \
    --quant-bf16 \
    vgg16_opt.mlir \
    -o vgg16_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter \
    vgg16_quant_bf16.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_bf16.bin \
    --dump-all-tensor=tensor_all_bf16.npz
bin_compare.py out.bin out_bf16.bin float32 1 1 1 1000 5
npz_compare.py tensor_all.npz tensor_all_bf16.npz

# VERDICT
echo $0 PASSED
