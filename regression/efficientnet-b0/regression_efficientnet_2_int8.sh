#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/efficientnet-b0.prototxt \
    --caffemodel $MODEL_PATH/caffe/efficientnet-b0.caffemodel \
    -o efficientnet-b0.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
   --convert-bn-to-scale \
   --fold-scale \
   --merge-scale-into-conv \
   efficientnet-b0.mlir \
   -o efficientnet-b0_opt.mlir

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/efficientnet-b0/data/efficientnet-b0_threshold_table \
    efficientnet-b0_opt.mlir \
    -o efficientnet-b0_cali.mlir

# quantization 1: per-layer int8
mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    efficientnet-b0_cali.mlir \
    -o efficientnet-b0_quant_int8_per_layer.mlir

# test mlir interpreter
mlir-tpu-interpreter efficientnet-b0_quant_int8_per_layer.mlir \
    --tensor-in $REGRESSION_PATH/efficientnet-b0/data/efficientnet_in_fp32.npz  \
    --tensor-out efficientnet_out_int8.npz \
    --dump-all-tensor=efficientnet_tensor_all_int8.npz 
# VERDICT
echo $0 PASSED
