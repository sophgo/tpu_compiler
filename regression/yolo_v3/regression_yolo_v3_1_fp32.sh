#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/yolov3/416/yolov3_416.prototxt \
    --caffemodel $MODEL_PATH/caffe/yolov3/416/yolov3_416.caffemodel \
    -o yolo_v3_416.mlir

# test mlir interpreter
mlir-tpu-interpreter yolo_v3_416.mlir \
    --tensor-in $DATA_PATH/test_dog_in_416x416_fp32.bin \
    --tensor-out out.bin \
    --dump-all-tensor=tensor_all.npz
bin_compare.py out.bin $DATA_PATH/test_dog_out_yolo_v3_416_fp32.bin \
    float32 1 1 1 689520

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    yolo_v3_416.mlir \
    -o yolo_v3_416_opt.mlir

# test opt
mlir-tpu-interpreter yolo_v3_416_opt.mlir \
    --tensor-in $DATA_PATH/test_dog_in_416x416_fp32.bin \
    --tensor-out out_opt.bin
bin_compare.py out_opt.bin out.bin float32 1 1 1 689520

# VERDICT
echo $0 PASSED
