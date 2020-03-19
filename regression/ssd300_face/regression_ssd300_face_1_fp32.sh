#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1
CHECK_NON_OPT_VERSION=0
# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/face_detection/ssd300_face/caffe/ssd300_face-deploy.prototxt \
    --caffemodel $MODEL_PATH/face_detection/ssd300_face/caffe/res10_300x300_ssd_iter_140000.caffemodel \
    -o ssd300_face.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_face_op_info.csv \
    ssd300_face.mlir \
    -o dummy.mlir

# test mlir interpreter
mlir-tpu-interpreter ssd300_face.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_fp32.npz \
    --dump-all-tensor=ssd300_face_tensor_all_fp32.npz

cvi_npz_tool.py compare ssd300_face_out_fp32.npz ssd300_face_out_fp32_ref.npz -v -d

cvi_npz_tool.py compare \
    ssd300_face_tensor_all_fp32.npz \
    ssd300_face_blobs.npz \
    --op_info ssd300_face_op_info.csv \
    --excepts detection_out \
    --tolerance=0.9999,0.9999,0.999 -vv
fi

mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --canonicalize \
    --tpu-op-info-filename ssd300_face_op_info.csv \
    ssd300_face.mlir \
    -o ssd300_face_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter ssd300_face_opt.mlir \
    --tensor-in ssd300_face_in_fp32.npz \
    --tensor-out ssd300_face_out_fp32.npz \
    --dump-all-tensor=ssd300_face_tensor_all_fp32.npz

cvi_npz_tool.py compare ssd300_face_out_fp32.npz ssd300_face_out_fp32_ref.npz -v -d

if [ $COMPARE_ALL -eq 1 ]; then
 cvi_npz_tool.py compare \
     ssd300_face_tensor_all_fp32.npz \
     ssd300_face_blobs.npz \
     --op_info ssd300_face_op_info.csv \
     --excepts detection_out \
     --tolerance=0.999,0.999,0.999 -vv
fi
# VERDICT
echo $0 PASSED
