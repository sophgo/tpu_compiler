#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# translate from caffe model
CHANNEL_ORDER=0,1,2
if [ $MODEL_CHANNEL_ORDER = "rgb" ]; then
    CHANNEL_ORDER=2,1,0
fi

cvi_model_convert.py \
      --model_path $MODEL_DEF \
      --model_dat $MODEL_DAT \
      --model_name ${NET} \
      --model_type $MODEL_TYPE \
      --batch_size $BATCH_SIZE \
      --raw_scale $RAW_SCALE \
      --mean $MEAN \
      --scale $INPUT_SCALE \
      --swap_channel $CHANNEL_ORDER \
      --mlir_file_path ${NET}.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    ${MLIR_OPT_FE_PRE} \
    --canonicalize \
    ${MLIR_OPT_FE_POST} \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    ${NET}.mlir \
    -o ${NET}_opt.mlir

# generate input data without preprocess
cvi_image_process.py \
    --image $IMAGE_PATH \
    --resize_dims $IMAGE_RESIZE_DIMS \
    --net_input_dims $NET_INPUT_DIMS \
    --batch $BATCH_SIZE \
    --save ${NET}_in_fp32.npz

mlir-tpu-interpreter ${NET}_opt.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor ${NET}_tensor_all_fp32.npz

# compare with caffe result
cvi_npz_tool.py compare \
    ${NET}_tensor_all_fp32.npz \
    ${NET}_blobs.npz \
    --op_info ${NET}_op_info.csv \
    --excepts $EXCEPTS \
    --tolerance=0.999,0.999,0.998 -vv

# VERDICT
echo $0 PASSED
