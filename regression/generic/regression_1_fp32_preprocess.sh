#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# translate from caffe model
if [ $MODEL_CHANNEL_ORDER = "rgb" ]; then
    mlir-translate \
        --caffe-to-mlir $MODEL_DEF \
        --caffemodel $MODEL_DAT \
        --resolve-preprocess \
        --raw_scale $RAW_SCALE \
        --mean $MEAN \
        --scale $INPUT_SCALE \
        --swap_channel 2,1,0 \
        --static-batchsize $BATCH_SIZE \
        -o ${NET}.mlir
else
    mlir-translate \
        --caffe-to-mlir $MODEL_DEF \
        --caffemodel $MODEL_DAT \
        --resolve-preprocess \
        --raw_scale $RAW_SCALE \
        --mean $MEAN \
        --scale $INPUT_SCALE \
        --swap_channel 0,1,2 \
        --static-batchsize $BATCH_SIZE \
        -o ${NET}.mlir
fi

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
    ${CUSTOM_OP_PLUGIN_OPTION} \
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
