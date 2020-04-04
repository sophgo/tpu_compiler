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

# function argument lower to MemRefType
mlir-opt \
    --debug \
    --convert-func-to-memref \
    retinaface_res50_with_detection_quant_int8_tg.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_memref.mlir

# op lower to MemRefType
mlir-opt \
    --debug \
    --convert-tg-op-to-memref \
    retinaface_res50_with_detection_quant_int8_tg_memref.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_op_memref.mlir

# memory space
mlir-opt \
    --debug \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref.csv \
    retinaface_res50_with_detection_quant_int8_tg_op_memref.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_op_memref_addr.mlir

# tg op back to TensorType
mlir-opt \
     --debug \
     --convert-tg-op-to-tensor \
    retinaface_res50_with_detection_quant_int8_tg_op_memref_addr.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_roundtrip.mlir

# function argument back to TensorType
mlir-opt \
    --debug \
    --convert-func-to-tensor \
    retinaface_res50_with_detection_quant_int8_tg_roundtrip.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_func_roundtrip.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_with_detection.csv \
    --tpu-weight-bin-filename=weight_int8_with_detection.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_with_detection_roundtrip.csv \
    retinaface_res50_with_detection_quant_int8_tg_func_roundtrip.mlir \
    -o retinaface_res50_with_detection_quant_int8_addr_roundtrip.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    retinaface_res50_with_detection_quant_int8_addr_roundtrip.mlir \
    -o cmdbuf_int8_with_detection_roundtrip.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_with_detection_roundtrip.bin \
    --weight weight_int8_with_detection.bin \
    --mlir retinaface_res50_with_detection_quant_int8_addr_roundtrip.mlir \
    --output=retinaface_res50_with_detection_int8_roundtrip.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input retinaface_res50_in_fp32.npz \
    --model retinaface_res50_with_detection_int8_roundtrip.cvimodel \
    --output retinaface_res50_with_detection_cmdbuf_out_all_int8_roundtrip.npz

# compare all tensors
cvi_npz_tool.py compare \
    retinaface_res50_with_detection_cmdbuf_out_all_int8_roundtrip.npz \
    retinaface_res50_tensor_all_int8.npz \
    --op_info retinaface_res50_with_detection_op_info_int8.csv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  mv retinaface_res50_with_detection_int8_roundtrip.cvimodel $CVIMODEL_REL_PATH
fi

#################
# Reuse global memory
#################
# memory space w/ reuse global memory 
mlir-opt \
    --debug \
    --enable-tpu-neuron-map-recyle-memref=1 \
    --assign-neuron-address-memref \
    --tpu-neuron-address-align-memref=16 \
    --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
    retinaface_res50_with_detection_quant_int8_tg_op_memref.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_op_memref_addr_reused.mlir

# tg op back to TensorType
mlir-opt \
     --debug \
     --convert-tg-op-to-tensor \
    retinaface_res50_with_detection_quant_int8_tg_op_memref_addr_reused.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_roundtrip_reused.mlir

# function argument back to TensorType
mlir-opt \
    --debug \
    --convert-func-to-tensor \
    retinaface_res50_with_detection_quant_int8_tg_roundtrip_reused.mlir \
    -o retinaface_res50_with_detection_quant_int8_tg_func_roundtrip_reused.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_with_detection.csv \
    --tpu-weight-bin-filename=weight_int8_with_detection.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_with_detection_roundtrip_reused.csv \
    retinaface_res50_with_detection_quant_int8_tg_func_roundtrip_reused.mlir \
    -o retinaface_res50_with_detection_quant_int8_addr_roundtrip_reused.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    retinaface_res50_with_detection_quant_int8_addr_roundtrip_reused.mlir \
    -o cmdbuf_int8_with_detection_roundtrip_reused.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_with_detection_roundtrip_reused.bin \
    --weight weight_int8_with_detection.bin \
    --mlir retinaface_res50_with_detection_quant_int8_addr_roundtrip_reused.mlir \
    --output=retinaface_res50_with_detection_int8_roundtrip_reused.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input retinaface_res50_in_fp32.npz \
    --model retinaface_res50_with_detection_int8_roundtrip_reused.cvimodel \
    --output retinaface_res50_with_detection_cmdbuf_out_all_int8_roundtrip_reused.npz

# compare all tensors
cvi_npz_tool.py compare \
    retinaface_res50_with_detection_cmdbuf_out_all_int8_roundtrip_reused.npz \
    retinaface_res50_tensor_all_int8.npz \
    --op_info retinaface_res50_with_detection_op_info_int8.csv

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  mv retinaface_res50_with_detection_int8_roundtrip_reused.cvimodel $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
