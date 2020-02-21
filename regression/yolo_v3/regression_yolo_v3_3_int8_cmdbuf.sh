#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=0

################################
# prepare int8 input
################################

npz_to_bin.py \
    yolo_v3_tensor_all_int8_multiplier.npz \
    data_quant \
    yolo_v3_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# npz_to_bin.py yolo_v3_in_fp32.npz input yolo_v3_in_fp32.bin
# bin_fp32_to_int8.py \
#     yolo_v3_in_fp32.bin \
#     yolo_v3_in_int8.bin \
#     1.0 \
#     1.00000488758

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    yolo_v3_416_quant_int8_per_layer.mlir \
    -o yolo_v3_416_quant_int8_per_layer_tg.mlir

# apply all possible backend optimizations
mlir-opt \
    --tg-fuse-leakyrelu \
    yolo_v3_416_quant_int8_per_layer_tg.mlir \
    -o yolo_v3_416_quant_int8_per_layer_tg_opt.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    yolo_v3_416_quant_int8_per_layer_tg_opt.mlir \
    -o yolo_v3_416_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    yolo_v3_416_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --neuron_map neuron_map.csv \
    --output=yolo_v3_416_int8_per_layer.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    yolo_v3_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    yolo_v3_cmdbuf_out_all_int8_per_layer.bin \
#    94614832 0 94614832 1
$RUNTIME_PATH/bin/test_cvinet \
    yolo_v3_in_int8.bin \
    yolo_v3_416_int8_per_layer.cvimodel \
    yolo_v3_cmdbuf_out_all_int8_per_layer.bin

bin_to_npz.py \
    yolo_v3_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    yolo_v3_cmdbuf_out_all_int8_per_layer.npz

npz_extract.py \
    yolo_v3_cmdbuf_out_all_int8_per_layer.npz \
    yolo_v3_out_int8_three_layer.npz \
    layer82-conv,layer94-conv,layer106-conv

npz_compare.py \
      yolo_v3_out_int8_three_layer.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --tolerance 0.98,0.88,0.79 -vv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_cmdbuf_out_all_int8_per_layer.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.90,0.88,0.51 -vv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)
fi

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

# apply all possible backend optimizations
mlir-opt \
    --tg-fuse-leakyrelu \
    yolo_v3_416_quant_int8_multiplier_tg.mlir \
    -o yolo_v3_416_quant_int8_multiplier_tg_opt.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    yolo_v3_416_quant_int8_multiplier_tg_opt.mlir \
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
#    94614832 0 94614832 1
$RUNTIME_PATH/bin/test_cvinet \
    yolo_v3_in_int8.bin \
    yolo_v3_416_int8_multiplier.cvimodel \
    yolo_v3_cmdbuf_out_all_int8_multiplier.bin

bin_to_npz.py \
    yolo_v3_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    yolo_v3_cmdbuf_out_all_int8_multiplier.npz

npz_extract.py \
    yolo_v3_cmdbuf_out_all_int8_multiplier.npz \
    yolo_v3_out_int8_multiplier_three_layer.npz \
    layer82-conv,layer94-conv,layer106-conv

npz_compare.py \
      yolo_v3_out_int8_multiplier_three_layer.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --tolerance 0.98,0.88,0.79 -vv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)

if [ $COMPARE_ALL -eq 1 ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_cmdbuf_out_all_int8_multiplier.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.90,0.88,0.51 -vv  # autotune-relu-overwrite-backward (with leakyrelu only neg quant)
fi
