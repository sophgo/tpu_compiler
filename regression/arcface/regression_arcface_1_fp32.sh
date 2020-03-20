#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.prototxt \
    --caffemodel $MODEL_PATH/face_recognition/arcface_res50/caffe/arcface_res50.caffemodel \
    -o arcface_res50.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename arcface_res50_op_info.csv \
      arcface_res50.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter arcface_res50.mlir \
      --tensor-in arcface_res50_in_fp32.npz \
      --tensor-out arcface_res50_out_fp32.npz \
      --dump-all-tensor=arcface_res50_tensor_all_fp32.npz
  cvi_npz_tool.py compare arcface_res50_out_fp32.npz arcface_res50_out_fp32_prob.npz -v
  cvi_npz_tool.py compare \
      arcface_res50_tensor_all_fp32.npz \
      arcface_res50_blobs.npz \
      --op_info arcface_res50_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi


# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename arcface_res50_op_info.csv \
    arcface_res50.mlir \
    -o arcface_res50_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter arcface_res50_opt.mlir \
    --tensor-in arcface_res50_in_fp32.npz \
    --tensor-out arcface_res50_out_fp32.npz \
    --dump-all-tensor=arcface_res50_tensor_all_fp32.npz

# bmface last layer is batchnorm, rename output
cvi_npz_tool.py rename arcface_res50_out_fp32.npz fc1_scale fc1
cvi_npz_tool.py compare arcface_res50_out_fp32.npz arcface_res50_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
      arcface_res50_tensor_all_fp32.npz \
      arcface_res50_blobs.npz \
      --op_info arcface_res50_op_info.csv \
      --tolerance=0.98,0.98,0.98 -vv


# VERDICT
echo $0 PASSED
