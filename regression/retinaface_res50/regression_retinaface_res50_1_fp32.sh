#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

RETINAFACE_USE_DECONV=0

if [ $RETINAFACE_USE_DECONV -eq 1 ]; then
  mlir-translate --caffe-to-mlir \
      $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt \
      --caffemodel $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel \
      -o retinaface_res50_deconv.mlir
else
  mlir-translate --caffe-to-mlir \
      $MODEL_PATH/face_detection/retinaface/caffe/R50-0000-upsample.prototxt \
      --caffemodel $MODEL_PATH/face_detection/retinaface/caffe/R50-0000-upsample.caffemodel \
      -o retinaface_res50.mlir
fi

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename retinaface_res50_op_info.csv \
      retinaface_res50.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter retinaface_res50.mlir \
      --tensor-in retinaface_res50_in_fp32.npz \
      --tensor-out retinaface_res50_out_fp32.npz \
      --dump-all-tensor=retinaface_res50_tensor_all_fp32.npz
  npz_compare.py retinaface_res50_out_fp32.npz retinaface_res50_out_fp32_caffe.npz -v
  npz_compare.py \
      retinaface_res50_tensor_all_fp32.npz \
      retinaface_res50_blobs.npz \
      --op_info retinaface_res50_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi

# apply all possible pre-calibration optimizations
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_res50_op_info.csv \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    --convert-scale-to-dwconv \
    --fuse-relu \
    retinaface_res50.mlir \
    -o retinaface_res50_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter \
    retinaface_res50_opt.mlir \
    --tensor-in retinaface_res50_in_fp32.npz \
    --tensor-out retinaface_res50_out_fp32.npz \
    --dump-all-tensor=retinaface_res50_tensor_all_fp32.npz

npz_compare.py retinaface_res50_out_fp32.npz retinaface_res50_out_fp32_caffe.npz -v
npz_compare.py \
    retinaface_res50_tensor_all_fp32.npz \
    retinaface_res50_caffe_blobs.npz \
    --op_info retinaface_res50_op_info.csv \
    --tolerance=0.999,0.999,0.999 -vvv

# VERDICT
echo $0 PASSED
