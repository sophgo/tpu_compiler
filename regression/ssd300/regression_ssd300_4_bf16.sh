#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

NEED_REMOVE_AFTER_FIX_CPU_LAYER=1

if [ $NEED_REMOVE_AFTER_FIX_CPU_LAYER -eq 1 ]; then

mlir-translate \
    --caffe-to-mlir $REGRESSION_PATH/ssd300/data/deploy_tpu.prototxt \
    --caffemodel $MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel \
    -o ssd300_bf16.mlir

mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --canonicalize \
    --tpu-op-info-filename ssd300_op_info.csv \
    ssd300_bf16.mlir \
    -o ssd300_quant_bf16_opt.mlir

#regenerate op info after opt for compare.
mlir-opt \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info.csv \
    ssd300_quant_bf16_opt.mlir

fi
# quantization
mlir-opt \
    --tpu-quant --quant-full-bf16 \
    --gen-sqrt-table \
    --gen-reciprocal-table \
    ssd300_quant_bf16_opt.mlir \
    -o ssd300_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter ssd300_quant_bf16.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_bf16.npz \
    --dump-all-tensor=ssd300_tensor_all_bf16.npz

cvi_npz_tool.py compare ssd300_out_bf16.npz ssd300_out_fp32.npz -v
cvi_npz_tool.py compare \
    ssd300_tensor_all_bf16.npz \
    ssd300_tensor_all_fp32.npz \
    --op_info ssd300_op_info.csv \
    --tolerance=0.99,0.99,0.90 -vvv

# VERDICT
echo $0 PASSED
