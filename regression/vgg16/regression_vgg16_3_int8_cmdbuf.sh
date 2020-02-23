#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py vgg16_in_fp32.npz input vgg16_in_fp32.bin
bin_fp32_to_int8.py \
    vgg16_in_fp32.bin \
    vgg16_in_int8.bin \
    1.0 \
    161.057006836

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
    vgg16_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --neuron_map neuron_map.csv \
    --output=vgg16_int8_per_layer.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    vgg16_in_int8.bin \
#    weight_int8_per_layer.bin \
#    cmdbuf_int8_per_layer.bin \
#    vgg16_cmdbuf_out_all_int8_per_layer.bin \
#    16460784 0 16460784 1
$RUNTIME_PATH/bin/test_cvinet \
    vgg16_in_int8.bin \
    vgg16_int8_per_layer.cvimodel \
    vgg16_cmdbuf_out_all_int8_per_layer.bin

bin_extract.py \
    vgg16_cmdbuf_out_all_int8_per_layer.bin \
    vgg16_cmdbuf_out_fc8_int8_per_layer.bin \
    int8 0x00024c00 1000
bin_compare.py \
    vgg16_cmdbuf_out_fc8_int8_per_layer.bin \
    $REGRESSION_PATH/vgg16/data/test_cat_out_vgg16_fc8_int8_per_layer.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    vgg16_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    vgg16_cmdbuf_out_all_int8_per_layer.npz
npz_compare.py \
    vgg16_cmdbuf_out_all_int8_per_layer.npz \
    vgg16_tensor_all_int8_per_layer.npz \
    --op_info vgg16_op_info_int8_per_layer.csv

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# quantization 3: multiplier int8
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    vgg16_quant_int8_multiplier.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --neuron_map neuron_map.csv \
    --output=vgg16_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    vgg16_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    vgg16_cmdbuf_out_all_int8_multiplier.bin \
#    16460784 0 16460784 1
$RUNTIME_PATH/bin/test_cvinet \
    vgg16_in_int8.bin \
    vgg16_int8_multiplier.cvimodel \
    vgg16_cmdbuf_out_all_int8_multiplier.bin

bin_extract.py \
    vgg16_cmdbuf_out_all_int8_multiplier.bin \
    vgg16_cmdbuf_out_fc8_int8_multiplier.bin \
    int8 0x00024c00 1000
bin_compare.py \
    vgg16_cmdbuf_out_fc8_int8_multiplier.bin \
    $REGRESSION_PATH/vgg16/data/test_cat_out_vgg16_fc8_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    vgg16_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    vgg16_cmdbuf_out_all_int8_multiplier.npz
npz_compare.py \
    vgg16_cmdbuf_out_all_int8_multiplier.npz \
    vgg16_tensor_all_int8_multiplier.npz \
    --op_info vgg16_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
