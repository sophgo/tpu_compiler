#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/mobilenet_v2_deploy.prototxt \
    --caffemodel $MODEL_PATH/caffe/mobilenet_v2.caffemodel \
    -o mobilenet_v2.mlir

# test mlir interpreter
mlir-tpu-interpreter mobilenet_v2.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
bin_compare.py out.bin $DATA_PATH/test_cat_out_mobilenet_v2_prob_fp32.bin \
    float32 1 1 1 1000 5 5

# opt1, convert bn to scale
mlir-opt \
    --convert-bn-to-scale \
    mobilenet_v2.mlir \
    -o mobilenet_v2_opt1.mlir

# test opt1
mlir-tpu-interpreter mobilenet_v2_opt1.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt1.bin
bin_compare.py out.bin out_opt1.bin float32 1 1 1 1000 5 5

# opt2, fold consecutive scales
mlir-opt \
    --fold-scale \
    mobilenet_v2_opt1.mlir \
    -o mobilenet_v2_opt2.mlir

# test opt2
mlir-tpu-interpreter mobilenet_v2_opt2.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt2.bin
bin_compare.py out.bin out_opt2.bin float32 1 1 1 1000 5 5

# opt3, merge scale into conv
mlir-opt \
    --merge-scale-into-conv \
    mobilenet_v2_opt2.mlir \
    -o mobilenet_v2_opt3.mlir

# test opt3
mlir-tpu-interpreter mobilenet_v2_opt3.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt3.bin
bin_compare.py out.bin out_opt3.bin float32 1 1 1 1000 5 5

# opt4, fuse relu into conv
mlir-opt \
    --fuse-relu \
    mobilenet_v2_opt3.mlir \
    -o mobilenet_v2_opt4.mlir

# test opt4
mlir-tpu-interpreter mobilenet_v2_opt4.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt4.bin
bin_compare.py out.bin out_opt4.bin float32 1 1 1 1000 5 5

# opt5, fuse eltwise with conv
mlir-opt \
    --fuse-eltwise \
    mobilenet_v2_opt4.mlir \
    -o mobilenet_v2_opt5.mlir

# test opt5
mlir-tpu-interpreter mobilenet_v2_opt5.mlir \
    --input-scale 0.017 \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt5.bin
bin_compare.py out.bin out_opt5.bin float32 1 1 1 1000 5 5

# VERDICT
echo $0 PASSED
