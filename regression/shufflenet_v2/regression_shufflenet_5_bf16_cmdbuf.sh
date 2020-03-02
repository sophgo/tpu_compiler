#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare bf16 input
################################
npz_to_bin.py shufflenet_in_fp32.npz input shufflenet_in_fp32.bin
bin_fp32_to_bf16.py \
    shufflenet_in_fp32.bin \
    shufflenet_in_bf16.bin

################################
# Lower
################################
mlir-opt \
    --tpu-lower \
    shufflenet_quant_bf16.mlir \
    -o shufflenet_quant_bf16_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map_bf16.csv \
    --tpu-weight-bin-filename=weight_bf16.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map_bf16.csv \
    shufflenet_quant_bf16_tg.mlir \
    -o shufflenet_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    shufflenet_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf_bf16.bin \
    --weight weight_bf16.bin \
    --neuron_map neuron_map_bf16.csv \
    --output=shufflenet_bf16.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    shufflenet_in_bf16.bin \
#    weight_bf16.bin \
#    cmdbuf_bf16.bin \
#    shufflenet_cmdbuf_out_all_bf16.bin \
#    32921552 0 32921552 1
test_cvinet \
    shufflenet_in_bf16.bin \
    shufflenet_bf16.cvimodel \
    shufflenet_cmdbuf_out_all_bf16.bin

bin_to_npz.py \
    shufflenet_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    shufflenet_cmdbuf_out_all_bf16.npz
npz_to_bin.py \
    shufflenet_cmdbuf_out_all_bf16.npz \
    fc \
    shufflenet_cmdbuf_out_fc_bf16.bin \
    bf16
bin_compare.py \
    shufflenet_cmdbuf_out_fc_bf16.bin \
    $REGRESSION_PATH/shufflenet_v2/data/test_cat_out_shufflenet_fc_bf16.bin \
    bf16 1 1 1 1000 5

# compare all tensors
npz_compare.py \
    shufflenet_cmdbuf_out_all_bf16.npz \
    shufflenet_tensor_all_bf16.npz \
    --op_info shufflenet_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vv

# VERDICT
echo $0 PASSED
