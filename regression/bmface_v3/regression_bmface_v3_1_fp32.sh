#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.prototxt \
    --caffemodel $MODEL_PATH/face_recognition/bmface/caffe/bmface-v3.caffemodel \
    -o bmface-v3.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename bmface-v3_op_info.csv \
      bmface-v3.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter bmface-v3.mlir \
      --tensor-in bmface-v3_in_fp32.npz \
      --tensor-out bmface-v3_out_fp32.npz \
      --dump-all-tensor=bmface-v3_tensor_all_fp32.npz
  npz_compare.py bmface-v3_out_fp32.npz bmface-v3_out_fp32_prob.npz -v
  npz_compare.py \
      bmface-v3_tensor_all_fp32.npz \
      bmface-v3_blobs.npz \
      --op_info bmface-v3_op_info.csv \
      --tolerance=0.9999,0.9999,0.999 -vv
fi


# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --convert-bn-to-scale \
    --canonicalize \
    --print-tpu-op-info \
    --tpu-op-info-filename bmface-v3_op_info.csv \
    bmface-v3.mlir \
    -o bmface-v3_opt.mlir

# test mlir interpreter
mlir-tpu-interpreter bmface-v3_opt.mlir \
    --tensor-in bmface-v3_in_fp32.npz \
    --tensor-out bmface-v3_out_fp32.npz \
    --dump-all-tensor=bmface-v3_tensor_all_fp32.npz

# bmface last layer is batchnorm, rename output
npz_rename.py bmface-v3_out_fp32.npz fc1_scale fc1
npz_compare.py bmface-v3_out_fp32.npz bmface-v3_out_fp32_prob.npz -v
npz_compare.py \
      bmface-v3_tensor_all_fp32.npz \
      bmface-v3_blobs.npz \
      --op_info bmface-v3_op_info.csv \
      --tolerance=0.98,0.98,0.98 -vv


# VERDICT
echo $0 PASSED
