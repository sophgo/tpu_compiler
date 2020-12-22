#!/bin/bash
set -xe

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

echo "$0 net=$NET"

if [ $MODEL_TYPE != "caffe" ]; then
  MODEL_DAT="-"
fi

# (x * (raw_scale/255) - mean ) / std * input_scale
cvi_model_convert.py \
    --model_path $MODEL_DEF \
    --model_dat $MODEL_DAT \
    --model_name ${NET} \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --net_input_dims ${NET_INPUT_DIMS} \
    --raw_scale ${RAW_SCALE} \
    --mean ${MEAN} \
    --std ${STD} \
    --input_scale ${INPUT_SCALE} \
    --model_channel_order $MODEL_CHANNEL_ORDER \
    --bgray ${BGRAY} \
    --mlir_file_path ${NET}.mlir

# assign layer_id right away, and apply all frontend optimizations
# Notes: convert-bn-to-scale has to be done before canonicalizer
tpuc-opt ${NET}.mlir \
    --convert-bn-to-scale \
    --convert-clip-to-relu6 \
    --canonicalize \
    --fuse-relu \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info.csv \
    -o ${NET}_opt_fp32.mlir

# test frontend optimizations
tpuc-interpreter ${NET}_opt_fp32.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_fp32.npz \
    --dump-all-tensor=${NET}_tensor_all_fp32.npz

if [ ${DO_POSTPROCESS} -eq 1 ]; then
  /bin/bash $POSTPROCESS_SCRIPT ${NET}_tensor_all_fp32.npz $OUTPUTS
fi

#cvi_npz_tool.py compare ${NET}_out_fp32.npz ${NET}_out_fp32_prob.npz -v

if [ "$NOT_COMPARE_FP32" != "1" ]; then
    cvi_npz_tool.py compare \
        ${NET}_tensor_all_fp32.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info.csv \
        --excepts $EXCEPTS \
        --tolerance=${TOLERANCE_FP32} -vv
fi

# VERDICT
echo $0 PASSED
