#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

mlir-translate --caffe-to-mlir \
    $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.prototxt \
    --caffemodel $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel \
    -o retinaface_res50.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_res50_op_info.csv \
    retinaface_res50.mlir \
    -o retinaface_res50_id.mlir

# apply all possible pre-calibration optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    retinaface_res50_id.mlir \
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
