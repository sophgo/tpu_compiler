#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"



# TODO: this is temporary, need to fix in interpreter directly
mlir-translate \
    --caffe-to-mlir $REGRESSION_PATH/ssd300/data/deploy_tpu.prototxt \
    --caffemodel $MODEL_PATH/object_detection/ssd/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel \
    -o ssd300.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info.csv \
    --canonicalize \
    ssd300.mlir \
    -o ssd300_opt.mlir

mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/ssd300/data/ssd300_threshold_table \
    ssd300_opt.mlir \
    -o ssd300_cali.mlir

###############################
#quantization 1: per-layer int8
###############################
mlir-opt \
    --tpu-quant --quant-int8-per-tensor \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info_int8_per_layer.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_per_layer.mlir

# ################################
# # quantization 3: per-channel multiplier int8
# ################################

mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info_int8_multiplier.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_multiplier.mlir