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

# apply all possible pre-calibration optimizations
# mlir-opt \
#     --convert-bn-to-scale \
#     --fold-scale \
#     --merge-scale-into-conv \
#     efficientnet-b0.mlir \
#     -o efficientnet-b0_opt.mlir

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table efficientnet-b0_threshold_table \
    efficientnet-b0.mlir \
    -o efficientnet-b0_cali.mlir

# # apply all possible post-calibration optimizations
# mlir-opt \
#     --fuse-relu \
#     --fuse-eltwise \
#     efficientnet-b0_cali.mlir \
#     -o efficientnet-b0_opt_post_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    efficientnet-b0_cali.mlir \
    -o efficientnet-b0_quant_int8_per_layer.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet-b0_quant_int8_per_layer.mlir \
    -debug-only=interpreter -debug \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
# VERDICT
echo $0 PASSED
