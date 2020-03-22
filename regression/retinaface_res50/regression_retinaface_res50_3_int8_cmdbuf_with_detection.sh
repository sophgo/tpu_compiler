#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


mlir-translate --caffe-to-mlir \
    $MODEL_PATH/face_detection/retinaface/caffe/R50-0000_with_detection.prototxt \
    --caffemodel $MODEL_PATH/face_detection/retinaface/caffe/R50-0000.caffemodel \
    -o retinaface_res50_with_detection.mlir

mlir-opt \
    --assign-layer-id \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_res50_op_info.csv \
    --convert-bn-to-scale \
    --canonicalize \
    retinaface_res50_with_detection.mlir \
    -o retinaface_res50_with_detection_opt.mlir

mlir-opt \
    --import-calibration-table \
    --calibration-table retinaface_res50_threshold_table \
    retinaface_res50_with_detection_opt.mlir \
    -o retinaface_res50_with_detection_cali.mlir

# Quantization
mlir-opt \
    --tpu-quant \
    --print-tpu-op-info \
    --tpu-op-info-filename retinaface_res50_with_detection_op_info_int8.csv \
    retinaface_res50_with_detection_cali.mlir \
    -o retinaface_res50_with_detection_quant_int8.mlir

mlir-opt \
    --tpu-lower \
    retinaface_res50_with_detection_quant_int8.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_with_detection.csv \
    --tpu-weight-bin-filename=weight_int8_with_detection.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_with_detection.csv \
    retinaface_res50_with_detection_quant_int8_tg.mlir \
    -o retinaface_res50_with_detection_quant_int8_addr.mlir

mlir-translate retinaface_res50_with_detection_quant_int8_addr.mlir \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_with_detection.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_with_detection.bin \
    --weight weight_int8_with_detection.bin \
    --mlir retinaface_res50_with_detection_quant_int8_addr.mlir \
    --output=retinaface_res50_with_detection_int8.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input retinaface_res50_in_fp32.npz \
    --model retinaface_res50_with_detection_int8.cvimodel \
    --output retinaface_res50_with_detection_cmdbuf_out_all_int8.npz

# compare all tensors
cvi_npz_tool.py compare \
    retinaface_res50_with_detection_cmdbuf_out_all_int8.npz \
    retinaface_res50_tensor_all_int8.npz \
    --op_info retinaface_res50_with_detection_op_info_int8.csv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  cp retinaface_res50_with_detection_int8.cvimodel $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
