#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/RGBIRlivenessFacebageNet.prototxt \
    --caffemodel $MODEL_PATH/RGBIRlivenessFacebageNet.caffemodel \
    -o liveness.mlir

# test mlir interpreter
mlir-tpu-interpreter liveness.mlir \
    --tensor-in $DIR/data/liveness_in_fp32.npz \
    --tensor-out liveness_out_fp32.npz \
    --dump-all-tensor=liveness_tensor_all_fp32.npz

npz_compare.py liveness_out_fp32.npz $DIR/data/liveness_out_fp32_fc2.npz -v

#npz_compare.py \
#    liveness_tensor_all_fp32.npz \
#    resnet50_blobs.npz \
#    --tolerance=0.9999,0.9999,0.999 -vvv

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    liveness.mlir \
    -o liveness_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter liveness_opt.mlir \
    --tensor-in $DIR/data/liveness_in_fp32.npz \
    --tensor-out liveness_opt_out_fp32.npz
npz_compare.py liveness_opt_out_fp32.npz $DIR/data/liveness_out_fp32_fc2.npz -v

# VERDICT
echo $0 PASSED
