#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh



mlir-translate \
    --caffe-to-mlir $REGRESSION_PATH/ssd300/data/deploy_tpu.prototxt \
    --caffemodel $MODEL_PATH/caffe/ssd300/VGG_coco_SSD_300x300_iter_400000.caffemodel \
    -o ssd300.mlir

# assign layer_id right away, and output op_info
mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info.csv \
    ssd300.mlir \
    -o ssd300_id.mlir

mlir-opt \
    --fuse-relu \
    ssd300_id.mlir \
    -o ssd300_opt1.mlir

#opt2, convert priorbox to loadweight 
mlir-opt \
    --convert-priorbox-to-loadweight \
    ssd300_opt1.mlir \
    -o ssd300_opt2.mlir

mlir-opt \
    --import-calibration-table \
    --calibration-table $REGRESSION_PATH/ssd300/data/ssd300_threshold_table \
    ssd300_opt2.mlir \
    -o ssd300_cali.mlir

###############################
#quantization 1: per-layer int8
###############################
mlir-opt \
    --quant-int8 \
    --print-tpu-op-info \
    --gen-sqrt-table \
    --gen-div-table \
    --tpu-op-info-filename ssd300_op_info_int8_per_layer.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_per_layer.mlir    

# ################################
# # quantization 3: per-channel multiplier int8
# ################################

mlir-opt \
    --quant-int8 \
    --enable-conv-per-channel \
    --enable-conv-multiplier \
    --gen-sqrt-table \
    --gen-div-table \
    --print-tpu-op-info \
    --tpu-op-info-filename ssd300_op_info_int8_multiplier.csv \
    ssd300_cali.mlir \
    -o ssd300_quant_int8_multiplier.mlir    