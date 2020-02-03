#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

TENSOR_IN_FILE=./data/bmface_v3_in_fp32_scale.npz

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/face_recognition/bmface/caffe/fp32/2020.01.15.01/bmface-v3.prototxt \
    --caffemodel $MODEL_PATH/face_recognition/bmface/caffe/fp32/2020.01.15.01/bmface-v3.caffemodel \
    -o bmface-v3.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename bmface-v3_op_info.csv \
    bmface-v3.mlir \
    -o bmface-v3_id.mlir

# test mlir interpreter
mlir-tpu-interpreter bmface-v3.mlir \
    --tensor-in $TENSOR_IN_FILE \
    --tensor-out bmface-v3_out_fp32.npz \
    --dump-all-tensor=bmface-v3_tensor_all_fp32.npz

# apply frontend optimizations
mlir-opt \
    --convert-bn-to-scale \
    --fold-scale \
    --merge-scale-into-conv \
    bmface-v3_id.mlir \
    -o bmface-v3_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter bmface-v3_opt.mlir \
    --tensor-in $TENSOR_IN_FILE \
    --tensor-out bmface-v3_opt_out_fp32.npz

#npz_compare.py bmface-v3_opt_out_fp32.npz bmface-v3_out_fp32_prob.npz -v
npz_compare.py bmface-v3_opt_out_fp32.npz bmface-v3_out_fp32.npz -v


# VERDICT
echo $0 PASSED
