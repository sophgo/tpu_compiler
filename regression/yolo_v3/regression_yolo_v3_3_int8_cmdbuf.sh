#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py yolo_v3_in_fp32.npz input yolo_v3_in_fp32.bin
bin_fp32_to_int8.py \
    yolo_v3_in_fp32.bin \
    yolo_v3_in_int8.bin \
    1.0 \
    1.00048852

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    yolo_v3_416_quant_int8_per_layer.mlir \
    -o yolo_v3_416_quant_int8_per_layer_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    yolo_v3_416_quant_int8_per_layer_tg.mlir \
    -o yolo_v3_416_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    yolo_v3_416_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    yolo_v3_in_int8.bin \
#    weight_int8_per_layer.bin \
#    cmdbuf_int8_per_layer.bin \
#    yolo_v3_cmdbuf_out_all_int8_per_layer.bin \
#    16460784 0 16460784 1
$RUNTIME_PATH/bin/test_cvinet \
    yolo_v3_in_int8.bin \
    yolo_v3_int8_per_layer.cvimodel \
    yolo_v3_cmdbuf_out_all_int8_per_layer.bin

bin_to_npz.py \
    yolo_v3_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    yolo_v3_cmdbuf_out_all_int8_per_layer.npz
npz_to_bin.py \
    yolo_v3_cmdbuf_out_all_int8_per_layer.npz \
    fc1000 \
    yolo_v3_cmdbuf_out_fc1000_int8_per_layer.bin \
    int8
bin_compare.py \
    yolo_v3_cmdbuf_out_fc1000_int8_per_layer.bin \
    $REGRESSION_PATH/yolo_v3/data/test_cat_out_yolo_v3_fc1000_int8_per_layer.bin \
    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    yolo_v3_cmdbuf_out_all_int8_per_layer.npz \
    yolo_v3_tensor_all_int8_per_layer.npz \
    --op_info yolo_v3_op_info_int8_per_layer.csv

################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    yolo_v3_416_quant_int8_multiplier.mlir \
    -o yolo_v3_416_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    yolo_v3_416_quant_int8_multiplier_tg.mlir \
    -o yolo_v3_416_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    yolo_v3_416_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --neuron_map neuron_map.csv \
    --output=yolo_v3_416_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    yolo_v3_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    yolo_v3_cmdbuf_out_all_int8_multiplier.bin \
#    16460784 0 16460784 1
$RUNTIME_PATH/bin/test_cvinet \
    yolo_v3_in_int8.bin \
    yolo_v3_int8_multiplier.cvimodel \
    yolo_v3_cmdbuf_out_all_int8_multiplier.bin

bin_to_npz.py \
    yolo_v3_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    yolo_v3_cmdbuf_out_all_int8_multiplier.npz
npz_to_bin.py \
    yolo_v3_cmdbuf_out_all_int8_multiplier.npz \
    fc1000 \
    yolo_v3_cmdbuf_out_fc1000_int8_multiplier.bin \
    int8
bin_compare.py \
    yolo_v3_cmdbuf_out_fc1000_int8_multiplier.bin \
    $REGRESSION_PATH/yolo_v3/data/test_cat_out_yolo_v3_fc1000_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    yolo_v3_cmdbuf_out_all_int8_multiplier.npz \
    yolo_v3_tensor_all_int8_multiplier.npz \
    --op_info yolo_v3_op_info_int8_multiplier.csv
