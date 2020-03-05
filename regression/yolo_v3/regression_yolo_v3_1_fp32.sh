#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.prototxt \
    --caffemodel $MODEL_PATH/object_detection/yolo_v3/caffe/416/yolov3_416.caffemodel \
    -o yolo_v3_416.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename yolo_v3_op_info.csv \
      yolo_v3_416.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter yolo_v3_416.mlir \
      --tensor-in yolo_v3_in_fp32.npz \
      --tensor-out yolo_v3_out_fp32.npz \
      --dump-all-tensor=yolo_v3_tensor_all_fp32.npz
  npz_compare.py yolo_v3_out_fp32.npz yolo_v3_out_fp32_ref.npz -v
  npz_compare.py \
      yolo_v3_tensor_all_fp32.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi

# assign layer_id right away, and apply all frontend optimizations
# Notes: convert-bn-to-scale has to be done before canonicalizer
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename yolo_v3_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    yolo_v3_416.mlir \
    -o yolo_v3_416_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter yolo_v3_416_opt.mlir \
    --tensor-in yolo_v3_in_fp32.npz \
    --tensor-out yolo_v3_out_fp32.npz \
    --dump-all-tensor=yolo_v3_tensor_all_fp32.npz
npz_compare.py yolo_v3_out_fp32.npz yolo_v3_out_fp32_ref.npz -v
npz_compare.py \
    yolo_v3_tensor_all_fp32.npz \
    yolo_v3_blobs.npz \
    --op_info yolo_v3_op_info.csv \
    --tolerance=0.999,0.999,0.999 -vv

# VERDICT
echo $0 PASSED
