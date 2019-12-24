#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
rm -rf *.npz
# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/efficientnet-b0.prototxt \
    --caffemodel $MODEL_PATH/caffe/efficientnet-b0.caffemodel \
    -debug \
    -o efficientnet-b0.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet-b0.mlir \
    -debug-only=interpreter -debug \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
# VERDICT
echo $0 PASSED
