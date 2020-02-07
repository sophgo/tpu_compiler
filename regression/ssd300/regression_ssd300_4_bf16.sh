#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

NEED_REMOVE_AFTER_FIX_CPU_LAYER=1

if [ $NEED_REMOVE_AFTER_FIX_CPU_LAYER -eq 1 ]; then

mlir-translate \
    --caffe-to-mlir $MODEL_PATH/object_detection/ssd/caffe/ssd300/deploy.prototxt \
    --caffemodel $MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel \
    -o ssd300.mlir
# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info.csv \
    ssd300.mlir \
    -o ssd300_id.mlir

# opt1, fuse relu with conv
mlir-opt \
    --fuse-relu \
    ssd300_id.mlir \
    -o ssd300_opt1.mlir
#opt2, convert priorbox to loadweight
mlir-opt \
    --convert-priorbox-to-loadweight \
    ssd300_opt1.mlir \
    -o ssd300_opt2.mlir

fi
# quantization
mlir-opt \
    --quant-bf16 \
    ssd300_opt2.mlir \
    -o ssd300_quant_bf16.mlir

# bf16 inference
mlir-tpu-interpreter ssd300_quant_bf16.mlir \
    --tensor-in ssd300_in_fp32.npz \
    --tensor-out ssd300_out_bf16.npz \
    --dump-all-tensor=ssd300_tensor_all_bf16.npz
npz_compare.py ssd300_out_bf16.npz ssd300_out_fp32.npz -v
npz_compare.py \
    ssd300_tensor_all_bf16.npz \
    ssd300_tensor_all_fp32.npz \
    --op_info ssd300_op_info.csv \
    --tolerance=0.99,0.99,0.91 -vvv

# VERDICT
echo $0 PASSED
