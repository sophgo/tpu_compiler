#!/bin/bash
set -e

#DIR="$( cd "$(dirname "$0")" ; pwd -P )"
#source $DIR/../../envsetup.sh

TENSOR_IN_FILE=./data/bmface_v3_in_fp32_scale.npz
TENSOR_IN_BIN_FILE=./data/bmface_v3_in_bf16_scale.bin

################################
# prepare bf16 input
################################
cvi_npz_tool.py to_bin $TENSOR_IN_FILE data ./data/bmface_v3_in_fp32_scale.bin
bin_fp32_to_bf16.py \
    ./data/bmface_v3_in_fp32_scale.bin \
    $TENSOR_IN_BIN_FILE

#################################
## quantization
#################################
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
    bmface_v3_quant_bf16.mlir \
    -o bmface_v3_quant_bf16_addr.mlir

# backend translate into cmdbuf
mlir-translate \
    --mlir-to-cmdbuf \
    bmface_v3_quant_bf16_addr.mlir \
    -o cmdbuf_bf16.bin

# run cmdbuf
$RUNTIME_PATH/bin/test_bmnet \
    $TENSOR_IN_BIN_FILE \
    weight_bf16.bin \
    cmdbuf_bf16.bin \
    bmface_v3_cmdbuf_out_all_bf16.bin \
    27975168 0 27975168 1

# (27975168 = 0x01925e00 + 2*64*112*112)

#bin_extract.py \
#    bmface_v3_cmdbuf_out_all_bf16.bin \
#    bmface_v3_cmdbuf_out_fc1000_bf16.bin \
#    bf16 0x00049800 1000
#bin_compare.py \
#    bmface_v3_cmdbuf_out_fc1000_bf16.bin \
#    $REGRESSION_PATH/bmface_v3/data/test_cat_out_bmface_v3_fc1000_bf16.bin \
#    bf16 1 1 1 1000 5

# compare all tensors
bin_to_npz.py \
    bmface_v3_cmdbuf_out_all_bf16.bin \
    neuron_map_bf16.csv \
    bmface_v3_cmdbuf_out_all_bf16.npz
cvi_npz_tool.py compare \
    bmface_v3_cmdbuf_out_all_bf16.npz \
    bmface_v3_tensor_all_bf16.npz \
    --op_info bmface_v3_op_info.csv \
    --tolerance=0.99,0.99,0.96 -vvv

# VERDICT
echo $0 PASSED
