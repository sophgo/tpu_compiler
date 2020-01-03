#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/ssd300/deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel \
    -o ssd300.mlir

if true; then

# test mlir interpreter
mlir-tpu-interpreter ssd300.mlir \
    --tensor-in $DATA_PATH/test_dog_in_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all.npz 

fi

# VERDICT
echo $0 PASSED
