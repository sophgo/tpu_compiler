#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# onnx inference only on python3, pls pip3 install onnxruntime
run_onnx_inference.py \
    --input_file $REGRESSION_PATH/data/cat.jpg \
    --mean ${MEAN} \
    --image_resize_dims ${IMAGE_RESIZE_DIMS} \
    --net_input_dims ${IMAGE_RESIZE_DIMS} \
    --std ${STD} \
    --raw_scale ${RAW_SCALE} \
    --output_file ${NET}_out_onnx.npz \
    --dump_tensor ${NET}_out_tensor_all.npz \
    --model_path $MODEL_DEF


cvi_npz_tool.py extract ${NET}_out_tensor_all.npz ${NET}_in_fp32.npz input
cp ${NET}_out_tensor_all.npz ${NET}_blobs.npz
# VERDICT
echo $0 PASSED
