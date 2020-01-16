#!/bin/bash

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/bmface-v3.prototxt \
    --caffemodel ./model/bmface-v3.caffemodel \
    -o ./model/bmface-v3.mlir


# pre-quantization optimization
mlir-opt  \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    bmface-v3.mlir \
    -o bmface-v3_opt.mlir


# run inference with mlir-tpu-interpreter
mlir-tpu-interpreter bmface-v3_opt.mlir \
    --tensor-in $DATA_PATH/Aaron_Eckhart_0001_112_112_fp32_scale.bin \
    --tensor-out out.bin


echo $0 PASSED
