#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

mlir-translate --caffe-to-mlir \
    $MODEL_PATH/face_detection/retinaface_mobilenet/caffe/190529/mnet_25.prototxt \
    --caffemodel $MODEL_PATH/face_detection/retinaface_mobilenet/caffe/190529/mnet_25.caffemodel \
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
  npz_compare.py retinaface_mnet25_out_fp32.npz retinaface_mnet25_out_fp32_caffe.npz -v
  npz_compare.py \
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
    --fold-scale \
    --merge-scale-into-conv \
    --convert-scale-to-dwconv \
    --fuse-relu \
    retinaface_mnet25.mlir \
    -o retinaface_mnet25_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter \
    retinaface_mnet25_opt.mlir \
    --tensor-in retinaface_mnet25_in_fp32.npz \
    --tensor-out retinaface_mnet25_out_fp32.npz \
    --dump-all-tensor=retinaface_mnet25_tensor_all_fp32.npz

npz_compare.py retinaface_mnet25_out_fp32.npz retinaface_mnet25_out_fp32_caffe.npz -v
npz_compare.py \
    retinaface_mnet25_tensor_all_fp32.npz \
    retinaface_mnet25_caffe_blobs.npz \
    --op_info retinaface_mnet25_op_info.csv \
    --tolerance=0.999,0.999,0.999 -vvv

# VERDICT
echo $0 PASSED
