#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CUSTOM_OP_PLUGIN_OPTION=""
if [[ ! -z $CUSTOM_OP_PLUGIN ]]; then
    CUSTOM_OP_PLUGIN_OPTION="--custom-op-plugin ${CUSTOM_OP_PLUGIN}"
fi

CHECK_NON_OPT_VERSION=0

if [ $MODEL_TYPE != "caffe" ]; then
    MODEL_DAT="-"
fi

if [ $DO_PREPROCESS -eq 1 ]; then
  # can't use caffe input directly
  # need to generate input data
  cvi_image_process.py \
      --image $IMAGE_PATH \
      --resize_dims $IMAGE_RESIZE_DIMS \
      --net_input_dims $NET_INPUT_DIMS \
      --batch $BATCH_SIZE \
      --yolo $YOLO_PREPROCESS \
      --save ${NET}_in_fp32.npz

  cvi_model_convert.py \
      --model_path $MODEL_DEF \
      --model_dat $MODEL_DAT \
      --model_name ${NET} \
      --model_type $MODEL_TYPE \
      --batch_size $BATCH_SIZE \
      --swap_channel $SWAP_CHANNEL \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --scale $INPUT_SCALE \
      --mlir_file_path ${NET}.mlir
else
  cvi_model_convert.py \
      --model_path $MODEL_DEF \
      --model_dat $MODEL_DAT \
      --model_name ${NET} \
      --model_type $MODEL_TYPE \
      --batch_size $BATCH_SIZE \
      --mlir_file_path ${NET}.mlir
fi

# assign layer_id right away, and apply all frontend optimizations
# Notes: convert-bn-to-scale has to be done before canonicalizer
mlir-opt \
    --fuse-relu \
    --assign-layer-id \
    ${MLIR_OPT_FE_PRE} \
    --canonicalize \
    ${MLIR_OPT_FE_POST} \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    ${NET}.mlir \
    -o ${NET}_opt.mlir

# test frontend optimizations
mlir-tpu-interpreter ${NET}_opt.mlir \
    ${CUSTOM_OP_PLUGIN_OPTION} \
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

if [ $DO_PREPROCESS -ne 1 ]; then
cvi_npz_tool.py to_bin ${NET}_in_fp32.npz $INPUT ${NET}_in_fp32.bin
fi

# VERDICT
echo $0 PASSED
