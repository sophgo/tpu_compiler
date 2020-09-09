#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

if [ $MODEL_TYPE != "caffe" ]; then
  MODEL_DAT="-"
fi

cvi_model_convert.py \
    --model_path $MODEL_DEF \
    --model_dat $MODEL_DAT \
    --model_name ${NET} \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --mlir_file_path ${NET}.mlir

# assign layer_id right away, and apply all frontend optimizations
# Notes: convert-bn-to-scale has to be done before canonicalizer
mlir-opt \
    ${MLIR_OPT_FE_PRE} \
    --canonicalize \
    ${MLIR_OPT_FE_POST} \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    ${NET}.mlir \
    -o ${NET}_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter ${NET}_opt.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz

#cvi_npz_tool.py compare ${NET}_out_fp32.npz ${NET}_out_fp32_prob.npz -v
cvi_npz_tool.py compare \
    ${NET}_tensor_all_fp32.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info.csv \
    --excepts $EXCEPTS \
    --tolerance=0.999,0.999,0.998 -vv

# VERDICT
echo $0 PASSED
