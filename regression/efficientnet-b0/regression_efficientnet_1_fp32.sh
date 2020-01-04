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
    
# apply opt 
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    efficientnet-b0.mlir \
    -o efficientnet-b0_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet-b0_opt.mlir \
    -debug-only=interpreter -debug \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
# VERDICT
echo $0 PASSED
