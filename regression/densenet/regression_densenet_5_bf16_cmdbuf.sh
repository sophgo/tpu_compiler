#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py densenet_in_fp32.npz input densenet_in_fp32.bin
bin_fp32_to_bf16.py \
    densenet_in_fp32.bin \
    densenet_in_bf16.bin

################################
# quantization
################################
# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    --assign-layer-id \
    densenet_quant_bf16.mlir \
    -o densenet_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    densenet_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --neuron_map neuron_map_bf16.csv \
    --output=densenet_bf16.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    densenet_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    densenet_cmdbuf_out_all_bf16.bin \
#    32921552 0 32921552 1
test_cvinet \
    densenet_in_bf16.bin \
    densenet_bf16.cvimodel \
    densenet_cmdbuf_out_all_bf16.bin

bin_extract.py \
    densenet_cmdbuf_out_all_bf16.bin \
    densenet_cmdbuf_out_fc6_bf16.bin \
    bf16 0x00049800 1000
bin_compare.py \
    densenet_cmdbuf_out_fc6_bf16.bin \
    $REGRESSION_PATH/densenet/data/cat_out_densenet_fc6_bf16.bin \
    bf16 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    densenet_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    densenet_cmdbuf_out_all_bf16.npz
npz_compare.py \
    densenet_cmdbuf_out_all_bf16.npz \
    densenet_tensor_all_bf16.npz \
    --op_info densenet_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vvv

# VERDICT
echo $0 PASSED
