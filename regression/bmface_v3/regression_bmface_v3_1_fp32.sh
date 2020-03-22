#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.prototxt \
    --caffemodel $MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.caffemodel \
    -o bmface_v3.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename bmface_v3_op_info.csv \
      bmface_v3.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter bmface_v3.mlir \
      --tensor-in bmface_v3_in_fp32.npz \
      --tensor-out bmface_v3_out_fp32.npz \
      --dump-all-tensor=bmface_v3_tensor_all_fp32.npz
  cvi_npz_tool.py compare bmface_v3_out_fp32.npz bmface_v3_out_fp32_prob.npz -v
  cvi_npz_tool.py compare \
      bmface_v3_tensor_all_fp32.npz \
      bmface_v3_blobs.npz \
      --op_info bmface_v3_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi


# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename bmface_v3_op_info.csv \
    bmface_v3.mlir \
    -o bmface_v3_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter bmface_v3_opt.mlir \
    --tensor-in bmface_v3_in_fp32.npz \
    --tensor-out bmface_v3_out_fp32.npz \
    --dump-all-tensor=bmface_v3_tensor_all_fp32.npz

# bmface last layer is batchnorm, rename output
cvi_npz_tool.py rename bmface_v3_out_fp32.npz fc1_scale fc1
cvi_npz_tool.py compare bmface_v3_out_fp32.npz bmface_v3_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
      bmface_v3_tensor_all_fp32.npz \
      bmface_v3_blobs.npz \
      --op_info bmface_v3_op_info.csv \
      --tolerance=0.98,0.98,0.98 -vv


# VERDICT
echo $0 PASSED
