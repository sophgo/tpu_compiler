#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py mobilenet_v2_in_fp32.npz input mobilenet_v2_in_fp32.bin
bin_fp32_to_int8.py \
    mobilenet_v2_in_fp32.bin \
    mobilenet_v2_in_int8.bin \
    1.0 \
    2.56929183

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
    mobilenet_v2_quant_int8_per_layer.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --neuron_map neuron_map.csv \
    --output=mobilenet_v2_int8_per_layer.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    mobilenet_v2_in_int8.bin \
#    weight_int8_per_layer.bin \
#    cmdbuf_int8_per_layer.bin \
#    mobilenet_v2_cmdbuf_out_all_int8_per_layer.bin \
#    9405584 0 9405584 1
test_cvinet \
    mobilenet_v2_in_int8.bin \
    mobilenet_v2_int8_per_layer.cvimodel \
    mobilenet_v2_cmdbuf_out_all_int8_per_layer.bin

bin_extract.py \
    mobilenet_v2_cmdbuf_out_all_int8_per_layer.bin \
    mobilenet_v2_cmdbuf_out_fc7_int8_per_layer.bin \
    int8 0x00024c00 1000
bin_compare.py \
    mobilenet_v2_cmdbuf_out_fc7_int8_per_layer.bin \
    $REGRESSION_PATH/mobilenet_v2/data/test_cat_out_mobilenet_v2_fc7_int8_per_layer.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    mobilenet_v2_cmdbuf_out_all_int8_per_layer.bin \
    neuron_map.csv \
    mobilenet_v2_cmdbuf_out_all_int8_per_layer.npz
npz_compare.py \
    mobilenet_v2_cmdbuf_out_all_int8_per_layer.npz \
    mobilenet_v2_tensor_all_int8_per_layer.npz \
    --op_info mobilenet_v2_op_info_int8_per_layer.csv

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
    mobilenet_v2_quant_int8_multiplier.mlir | \
  mlir-translate \
    --mlir-to-cmdbuf \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --neuron_map neuron_map.csv \
    --output=mobilenet_v2_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    mobilenet_v2_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    mobilenet_v2_cmdbuf_out_all_int8_multiplier.bin \
#    9405584 0 9405584 1
test_cvinet \
    mobilenet_v2_in_int8.bin \
    mobilenet_v2_int8_multiplier.cvimodel \
    mobilenet_v2_cmdbuf_out_all_int8_multiplier.bin

bin_extract.py \
    mobilenet_v2_cmdbuf_out_all_int8_multiplier.bin \
    mobilenet_v2_cmdbuf_out_fc7_int8_multiplier.bin \
    int8 0x00024c00 1000
bin_compare.py \
    mobilenet_v2_cmdbuf_out_fc7_int8_multiplier.bin \
    $REGRESSION_PATH/mobilenet_v2/data/test_cat_out_mobilenet_v2_fc7_int8_multiplier.bin \
    int8 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    mobilenet_v2_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    mobilenet_v2_cmdbuf_out_all_int8_multiplier.npz
npz_compare.py \
    mobilenet_v2_cmdbuf_out_all_int8_multiplier.npz \
    mobilenet_v2_tensor_all_int8_multiplier.npz \
    --op_info mobilenet_v2_op_info_int8_multiplier.csv

# VERDICT
echo $0 PASSED
