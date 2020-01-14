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


# calibration
mlir-opt \
    --import-calibration-table \
    --calibration-table bmface-v3_cali1024_threshold_table \
    bmface-v3_opt.mlir \
    -o bmface-v3_cali.mlir

## post-calibration optimization
## Not Work, fuse-eltwise fail.
## So I skip this step.
#mlir-opt \
#    -debug \
#    --fuse-relu \
#    --fuse-eltwise \
#    bmface_v2_cali.mlir \
#    -o bmface_v2_opt_post_cali.mlir

## quantization
## Can Work, but the precision of the interpreter result is very poor...
#mlir-opt \
#    --quant-int8 \
#    --enable-conv-multiplier \
#    --enable-conv-per-channel \
#    bmface-v3_cali.mlir \
#    -o bmface-v3_int8_conv_multiplier.mlir
#
## run int8 inference with mlir-tpu-interpreter
#mlir-tpu-interpreter bmface-v3_int8_conv_multiplier.mlir \
#    --tensor-in $DATA_PATH/Aaron_Eckhart_0001_112_112_fp32_scale.bin \
#    --tensor-out out.bin

echo $0 PASSED
