#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh
echo $0 IS RUNNING

################################
# prepare int8 input
################################
npz_to_bin.py inception_v3_in_fp32.npz input inception_v3_in_fp32.bin
bin_fp32_to_int8.py \
    inception_v3_in_fp32.bin \
    inception_v3_in_int8.bin \
    1.0 \
    151.133789062

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
    inception_v3_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    inception_v3_in_int8.bin \
    weight_int8_per_layer.bin \
    cmdbuf_int8_per_layer.bin \
    inception_v3_cmdbuf_out_all_int8_per_layer.bin \
    14422816 0 14422816 1
#bin_extract.py \
#    inception_v3_cmdbuf_out_all_int8_per_layer.bin \
#    inception_v3_cmdbuf_out_classifier_int8_per_layer.bin \
#    int8 0x000417b0 1000
#bin_compare.py \
#    inception_v3_cmdbuf_out_classifier_int8_per_layer.bin \
#    $REGRESSION_PATH/inception_v3/data/test_cat_out_inception_v3_classifier_int8_per_layer.bin \
#    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    inception_v3_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    inception_v3_cmdbuf_out_all_int8_per_layer.npz
npz_compare.py \
    inception_v3_cmdbuf_out_all_int8_per_layer.npz \
    inception_v3_tensor_all_int8_per_layer.npz \
    --order neuron_map.csv

################################
# quantization 2: per-channel int8
################################

# skipped

################################
# quantization 3: per-channel multiplier int8
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
    inception_v3_quant_int8_multiplier.mlir | \
mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_multiplier.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    inception_v3_in_int8.bin \
    weight_int8_multiplier.bin \
    cmdbuf_int8_multiplier.bin \
    inception_v3_cmdbuf_out_all_int8_multiplier.bin \
    14422816 0 14422816 1
#bin_extract.py \
#    inception_v3_cmdbuf_out_all_int8_multiplier.bin \
#    inception_v3_cmdbuf_out_classifier_int8_multiplier.bin \
#    int8 0x000417b0 1000
#bin_compare.py \
#    inception_v3_cmdbuf_out_classifier_int8_multiplier.bin \
#    $REGRESSION_PATH/inception_v3/data/test_cat_out_inception_v3_classifier_int8_multiplier.bin \
#    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    inception_v3_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    inception_v3_cmdbuf_out_all_int8_multiplier.npz
npz_compare.py \
    inception_v3_cmdbuf_out_all_int8_multiplier.npz \
    inception_v3_tensor_all_int8_multiplier.npz \
    --order neuron_map.csv

# VERDICT
echo $0 PASSED
