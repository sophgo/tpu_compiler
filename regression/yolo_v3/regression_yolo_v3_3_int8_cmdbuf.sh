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
# quantization 1: per-layer int8
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    yolo_v3_416_quant_int8_per_layer_fused_relu.mlir  | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    yolo_v3_in_int8.bin \
    weight_int8_per_layer.bin \
    cmdbuf_int8_per_layer.bin \
    yolo_v3_cmdbuf_out_all_int8_per_layer.bin \
    56326192 0 56326192 1

bin_to_npz.py \
    yolo_v3_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    yolo_v3_cmdbuf_out_all_int8_per_layer.npz

npz_extract.py \
    yolo_v3_cmdbuf_out_all_int8_per_layer.npz \
    yolo_v3_cmdbuf_out_three_layer_result.npz \
    layer82-conv,layer94-conv,layer106-conv

npz_compare.py \
      yolo_v3_cmdbuf_out_three_layer_result.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --tolerance 0.9,0.85,0.75 -vvv

if [ $COMPARE_ALL ]; then
  # some tensors do not pass due to threshold bypass
  # need do dequantization in interpreter directly
  npz_compare.py \
      yolo_v3_cmdbuf_out_all_int8_per_layer.npz \
      yolo_v3_blobs.npz \
      --op_info yolo_v3_op_info_int8_per_layer.csv \
      --dequant \
      --excepts layer86-upsample,layer87-route,layer98-upsample,layer99-route \
      --tolerance 0.75,0.7,0.1 -vvv
fi