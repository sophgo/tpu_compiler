#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --fuse-scale-into-conv \
    --fuse-relu \
    mobilenet_v2.mlir \
    -o mobilenet_v2_opt.mlir

# fp32 inference
mlir-tpu-interpreter mobilenet_v2_opt.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all.npz

# quantization
mlir-opt \
    --quant-bf16 \
    mobilenet_v2_opt.mlir \
    -o mobilenet_v2_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter \
    mobilenet_v2_quant_bf16.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_bf16.bin \
    --dump-all-tensor=tensor_all_bf16.npz
bin_compare.py out.bin out_bf16.bin float32 1 1 1 1000 5
npz_compare.py tensor_all.npz tensor_all_bf16.npz

# VERDICT
echo $0 PASSED
