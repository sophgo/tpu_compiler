#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.prototxt \
    --caffemodel $MODEL_PATH/face_antispoofing/RGBIRLiveness/caffe/RGBIRlivenessFacebageNet.caffemodel \
    -o liveness.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename resnet50_op_info.csv \
      liveness.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter liveness.mlir \
      --tensor-in liveness_in_fp32.npz \
      --tensor-out liveness_out_fp32.npz \
      --dump-all-tensor=liveness_tensor_all_fp32.npz
  cvi_npz_tool.py compare liveness_out_fp32.npz resnet50_out_fp32_prob.npz -v
  cvi_npz_tool.py compare \
      liveness_tensor_all_fp32.npz \
      liveness_blobs.npz \
      --op_info liveness_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi

# apply frontend optimizations
mlir-opt \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename liveness_op_info.csv \
    liveness.mlir \
    -o liveness_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter liveness_opt.mlir \
    --tensor-in liveness_in_fp32.npz \
    --tensor-out liveness_opt_out_fp32.npz \
    --dump-all-tensor=liveness_tensor_all_fp32.npz

cvi_npz_tool.py compare liveness_opt_out_fp32.npz liveness_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
   liveness_tensor_all_fp32.npz \
   liveness_blobs.npz \
   --op_info liveness_op_info.csv \
   --tolerance=0.9999,0.9999,0.999 -vv

# VERDICT
echo $0 PASSED
