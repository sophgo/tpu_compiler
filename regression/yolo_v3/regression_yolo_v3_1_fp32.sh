#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/caffe/yolov3/416/yolov3_416.prototxt \
    --caffemodel $MODEL_PATH/caffe/yolov3/416/yolov3_416.caffemodel \
    -o yolo_v3_416.mlir

if false; then

# test mlir interpreter
mlir-tpu-interpreter yolo_v3_416.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out.bin
bin_compare.py out.bin $DATA_PATH/test_cat_out_yolo_v3_416_prob_fp32.bin \
    float32 1 1 1 1000 5 5

# opt1, convert bn to scale
mlir-opt \
    --convert-bn-to-scale \
    yolo_v3_416.mlir \
    -o yolo_v3_416_opt1.mlir

# test opt1
mlir-tpu-interpreter yolo_v3_416_opt1.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt1.bin
bin_compare.py out.bin out_opt1.bin float32 1 1 1 1000 5 5

# opt2, fold consecutive scales
mlir-opt \
    --fold-scale \
    yolo_v3_416_opt1.mlir \
    -o yolo_v3_416_opt2.mlir

# test opt2
mlir-tpu-interpreter yolo_v3_416_opt2.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt2.bin
bin_compare.py out.bin out_opt2.bin float32 1 1 1 1000 5 5

# opt3, merge scale into conv
mlir-opt \
    --merge-scale-into-conv \
    yolo_v3_416_opt2.mlir \
    -o yolo_v3_416_opt3.mlir

# test opt3
mlir-tpu-interpreter yolo_v3_416_opt3.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt3.bin
bin_compare.py out.bin out_opt3.bin float32 1 1 1 1000 5 5

# opt4, fuse relu with conv
mlir-opt \
    --fuse-relu \
    yolo_v3_416_opt3.mlir \
    -o yolo_v3_416_opt4.mlir

# test opt4
mlir-tpu-interpreter yolo_v3_416_opt4.mlir \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt4.bin
bin_compare.py out.bin out_opt4.bin float32 1 1 1 1000 5 5

# opt5, fuse eltwise with conv
mlir-opt \
    --fuse-eltwise \
    yolo_v3_416_opt4.mlir \
    -o yolo_v3_416_opt5.mlir

# test opt5
mlir-tpu-interpreter yolo_v3_416_opt5.mlir \
    --tpu-op-stats-filename yolo_v3_416_op_stats.csv \
    --tensor-in $DATA_PATH/test_cat_in_fp32.bin \
    --tensor-out out_opt5.bin
bin_compare.py out.bin out_opt5.bin float32 1 1 1 1000 5 5

fi

# VERDICT
echo $0 PASSED
