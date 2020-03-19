#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

CHECK_NON_OPT_VERSION=0

# translate from caffe model
mlir-translate \
    --caffe-to-mlir $MODEL_DEF \
    --caffemodel $MODEL_DAT \
    --static-batchsize $BATCH_SIZE \
    -o ${NET}.mlir

if [ $CHECK_NON_OPT_VERSION -eq 1 ]; then
  mlir-opt \
      --assign-layer-id \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info.csv \
      ${NET}.mlir \
      -o dummy.mlir
  # test mlir interpreter
  mlir-tpu-interpreter ${NET}.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_fp32.npz \
      --dump-all-tensor=${NET}_tensor_all_fp32.npz
  cvi_npz_tool.py compare ${NET}_out_fp32.npz ${NET}_out_fp32_prob.npz -v
  cvi_npz_tool.py compare \
      ${NET}_tensor_all_fp32.npz \
      ${NET}_blobs.npz \
      --op_info ${NET}_op_info.csv \
      --tolerance=0.9999,0.999,0.999 -vv
fi

# assign layer_id right away, and apply all frontend optimizations
# Notes: convert-bn-to-scale has to be done before canonicalizer
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    ${NET}.mlir \
    -o ${NET}_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter ${NET}_opt.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz
cvi_npz_tool.py compare ${NET}_out_fp32.npz ${NET}_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    ${NET}_tensor_all_fp32.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info.csv \
    --tolerance=0.9999,0.999,0.999 -vv

# VERDICT
echo $0 PASSED
