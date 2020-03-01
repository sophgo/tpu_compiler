#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

################################
# prepare int8 input
################################

npz_to_bin.py \
    inception_v4_tensor_all_int8_multiplier.npz \
    data \
    inception_v4_in_int8.bin \
    int8

# npz_to_bin.py inception_v4_in_fp32.npz input inception_v4_in_fp32.bin
# bin_fp32_to_int8.py \
#     inception_v4_in_fp32.bin \
#     inception_v4_in_int8.bin \
#     1.0 \
#     0.994933307171

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    inception_v4_quant_int8_per_layer.mlir \
    -o inception_v4_quant_int8_per_layer_tg.mlir

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
    inception_v4_quant_int8_per_layer_tg.mlir  \
    -o inception_v4_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    inception_v4_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --neuron_map neuron_map.csv \
    --output=inception_v4_int8_per_layer.cvimodel

## run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    inception_v4_in_int8.bin \
#    weight_int8_per_layer.bin \
#    cmdbuf_int8_per_layer.bin \
#    inception_v4_cmdbuf_out_all_int8_per_layer.bin \
#    27293984 0 27293984 1

$RUNTIME_PATH/bin/test_cvinet \
    inception_v4_in_int8.bin \
    inception_v4_int8_per_layer.cvimodel \
    inception_v4_cmdbuf_out_all_int8_per_layer.bin

bin_to_npz.py \
    inception_v4_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    inception_v4_cmdbuf_out_all_int8_per_layer.npz
npz_to_bin.py \
    inception_v4_cmdbuf_out_all_int8_per_layer.npz \
    classifier \
    inception_v4_cmdbuf_out_classifier_int8_per_layer.bin \
    int8
#bin_compare.py \
#    inception_v4_cmdbuf_out_classifier_int8_per_layer.bin \
#    $REGRESSION_PATH/inception_v4/data/test_cat_out_inception_v4_classifier_int8_per_layer.bin \
#    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    inception_v4_cmdbuf_out_all_int8_per_layer.npz \
    inception_v4_tensor_all_int8_per_layer.npz \
    --op_info inception_v4_op_info_int8_per_layer.csv

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    inception_v4_quant_int8_multiplier.mlir \
    -o inception_v4_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    inception_v4_quant_int8_multiplier_tg.mlir \
    -o inception_v4_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    inception_v4_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --neuron_map neuron_map.csv \
    --output=inception_v4_int8_multiplier.cvimodel

## run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    inception_v4_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    inception_v4_cmdbuf_out_all_int8_multiplier.bin \
#    27293984 0 27293984 1
$RUNTIME_PATH/bin/test_cvinet \
    inception_v4_in_int8.bin \
    inception_v4_int8_multiplier.cvimodel \
    inception_v4_cmdbuf_out_all_int8_multiplier.bin

bin_to_npz.py \
    inception_v4_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    inception_v4_cmdbuf_out_all_int8_multiplier.npz
npz_to_bin.py \
    inception_v4_cmdbuf_out_all_int8_multiplier.npz \
    classifier \
    inception_v4_cmdbuf_out_classifier_int8_multiplier.bin \
    int8
#bin_compare.py \
#    inception_v4_cmdbuf_out_classifier_int8_multiplier.bin \
#    $REGRESSION_PATH/inception_v4/data/test_cat_out_inception_v4_classifier_int8_multiplier.bin \
#    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    inception_v4_cmdbuf_out_all_int8_multiplier.npz \
    inception_v4_tensor_all_int8_multiplier.npz \
    --op_info inception_v4_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
