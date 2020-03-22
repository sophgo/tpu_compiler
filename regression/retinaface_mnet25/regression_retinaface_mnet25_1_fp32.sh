#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


CHECK_NON_OPT_VERSION=0

mlir-translate --caffe-to-mlir \
    $MODEL_PATH/face_detection/retinaface/caffe/mnet_320.prototxt \
    --caffemodel $MODEL_PATH/face_detection/retinaface/caffe/mnet.caffemodel \
    -o retinaface_mnet25.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename retinaface_mnet25_op_info.csv \
      retinaface_mnet25.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter retinaface_mnet25.mlir \
      --tensor-in retinaface_mnet25_in_fp32.npz \
      --tensor-out retinaface_mnet25_out_fp32.npz \
      --dump-all-tensor=retinaface_mnet25_tensor_all_fp32.npz
  cvi_npz_tool.py compare retinaface_mnet25_out_fp32.npz retinaface_mnet25_out_fp32_caffe.npz -v
  cvi_npz_tool.py compare \
      retinaface_mnet25_tensor_all_fp32.npz \
      retinaface_mnet25_blobs.npz \
      --op_info retinaface_mnet25_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi

# apply all possible pre-calibration optimizations
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_mnet25_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    retinaface_mnet25.mlir \
    -o retinaface_mnet25_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter \
    retinaface_mnet25_opt.mlir \
    --tensor-in retinaface_mnet25_in_fp32.npz \
    --tensor-out retinaface_mnet25_out_fp32.npz \
    --dump-all-tensor=retinaface_mnet25_tensor_all_fp32.npz

cvi_npz_tool.py compare retinaface_mnet25_out_fp32.npz retinaface_mnet25_out_fp32_caffe.npz -v
cvi_npz_tool.py compare \
    retinaface_mnet25_tensor_all_fp32.npz \
    retinaface_mnet25_caffe_blobs.npz \
    --op_info retinaface_mnet25_op_info.csv \
    --tolerance=0.999,0.999,0.999 -vvv

# VERDICT
echo $0 PASSED
