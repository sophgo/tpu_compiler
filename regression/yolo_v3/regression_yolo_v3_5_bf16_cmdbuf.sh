#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py yolo_v3_in_fp32.npz input yolo_v3_in_fp32.bin
bin_fp32_to_bf16.py \
    yolo_v3_in_fp32.bin \
    yolo_v3_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    yolo_v3_416_quant_bf16.mlir \
    -o yolo_v3_416_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    yolo_v3_416_quant_bf16_tg.mlir \
    -o yolo_v3_416_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    yolo_v3_416_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --neuron_map neuron_map_bf16.csv \
    --output=yolo_v3_416_bf16.cvimodel

# run cmdbuf
test_cvinet \
    yolo_v3_in_bf16.bin \
    yolo_v3_416_bf16.cvimodel \
    yolo_v3_416_cmdbuf_out_all_bf16.bin

bin_to_npz.py \
    yolo_v3_416_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    yolo_v3_416_cmdbuf_out_all_bf16.npz

npz_extract.py \
    yolo_v3_416_cmdbuf_out_all_bf16.npz \
    yolo_v3_out_bf16_three_layer.npz \
    layer82-conv,layer94-conv,layer106-conv

npz_compare.py \
      yolo_v3_out_bf16_three_layer.npz \
      yolo_v3_416_tensor_all_bf16.npz \
      --op_info yolo_v3_op_info_bf16_per_layer.csv