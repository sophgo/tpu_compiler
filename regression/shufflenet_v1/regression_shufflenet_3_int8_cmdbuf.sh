#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################

npz_to_bin.py \
    shufflenet_tensor_all_int8_multiplier.npz \
    data_quant \
    shufflenet_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# npz_to_bin.py shufflenet_in_fp32.npz input shufflenet_in_fp32.bin
# bin_fp32_to_int8.py \
#    shufflenet_in_fp32.bin \
#    shufflenet_in_int8.bin \
#    1.0 \
#    161.008057

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    shufflenet_quant_int8_per_layer.mlir \
    -o shufflenet_quant_int8_per_layer_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    shufflenet_quant_int8_per_layer_tg.mlir \
    -o shufflenet_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    shufflenet_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --neuron_map neuron_map.csv \
    --output=shufflenet_int8_per_layer.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    shufflenet_in_int8.bin \
#    weight_int8_per_layer.bin \
#    cmdbuf_int8_per_layer.bin \
#    shufflenet_cmdbuf_out_all_int8_per_layer.bin \
#    16460784 0 16460784 1
$RUNTIME_PATH/bin/test_cvinet \
    shufflenet_in_int8.bin \
    shufflenet_int8_per_layer.cvimodel \
    shufflenet_cmdbuf_out_all_int8_per_layer.bin

bin_to_npz.py \
    shufflenet_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    shufflenet_cmdbuf_out_all_int8_per_layer.npz
npz_to_bin.py \
    shufflenet_cmdbuf_out_all_int8_per_layer.npz \
    fc1000 \
    shufflenet_cmdbuf_out_fc1000_int8_per_layer.bin \
    int8
bin_compare.py \
    shufflenet_cmdbuf_out_fc1000_int8_per_layer.bin \
    $REGRESSION_PATH/shufflenet/data/test_cat_out_shufflenet_fc1000_int8_per_layer.bin \
    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    shufflenet_cmdbuf_out_all_int8_per_layer.npz \
    shufflenet_tensor_all_int8_per_layer.npz \
    --op_info shufflenet_op_info_int8_per_layer.csv

################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    shufflenet_quant_int8_multiplier.mlir \
    -o shufflenet_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    shufflenet_quant_int8_multiplier_tg.mlir \
    -o shufflenet_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    shufflenet_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
python $CVIBUILDER_PATH/python/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --neuron_map neuron_map.csv \
    --output=shufflenet_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    shufflenet_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    shufflenet_cmdbuf_out_all_int8_multiplier.bin \
#    16460784 0 16460784 1
$RUNTIME_PATH/bin/test_cvinet \
    shufflenet_in_int8.bin \
    shufflenet_int8_multiplier.cvimodel \
    shufflenet_cmdbuf_out_all_int8_multiplier.bin

bin_to_npz.py \
    shufflenet_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    shufflenet_cmdbuf_out_all_int8_multiplier.npz
npz_to_bin.py \
    shufflenet_cmdbuf_out_all_int8_multiplier.npz \
    fc1000 \
    shufflenet_cmdbuf_out_fc1000_int8_multiplier.bin \
    int8
bin_compare.py \
    shufflenet_cmdbuf_out_fc1000_int8_multiplier.bin \
    $REGRESSION_PATH/shufflenet/data/test_cat_out_shufflenet_fc1000_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    shufflenet_cmdbuf_out_all_int8_multiplier.npz \
    shufflenet_tensor_all_int8_multiplier.npz \
    --op_info shufflenet_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
