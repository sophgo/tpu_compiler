#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

# create int8 input
npz_to_bin.py \
    efficientnet_tensor_all_int8.npz \
    data \
    efficientnet_in_int8.bin \
    int8
# npz_to_bin.py efficientnet_in_fp32.npz data efficientnet_in_fp32.bin
# bin_fp32_to_int8.py \
#     efficientnet_in_fp32.bin \
#     efficientnet_in_int8.bin \
#     1.0 \
#     2.64064478874

#  Lower for quantization
mlir-opt \
    --tpu-lower \
    efficientnet_b0_quant_int8_multiplier.mlir \
    -o efficientnet_b0_quant_int8_multiplier_tg.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    --assign-layer-id \
    efficientnet_b0_quant_int8_multiplier_tg.mlir \
    -o  efficientnet_b0_quant_int8_multiplier_cmdbuf.mlir

mlir-translate \
    efficientnet_b0_quant_int8_multiplier_cmdbuf.mlir \
     --mlir-to-cmdbuf \
     -o cmdbuf.bin

# generate cvi model
python $TPU_PYTHON_PATH/cvi_model_create.py \
    --cmdbuf cmdbuf.bin \
    --weight weight.bin \
    --neuron_map neuron_map.csv \
    --output=efficientnet_int8_multiplier.cvimodel

# run cmdbuf
test_cvinet \
    efficientnet_in_int8.bin \
    efficientnet_int8_multiplier.cvimodel \
    efficientnet_cmdbuf_out_all_int8_multiplier.bin

bin_to_npz.py \
    efficientnet_cmdbuf_out_all_int8_multiplier.bin \
    neuron_map.csv \
    efficientnet_cmdbuf_out_all_int8_multiplier.npz

# run interpreter, to generate reference tensor all npz
# mlir-tpu-interpreter efficientnet_b0_quant_int8_multiplier.mlir \
#     --tensor-in efficientnet_in_fp32.npz  \
#     --tensor-out efficientnet_out_int8.npz \
#     --dump-all-tensor=efficientnet_tensor_all_int8.npz

# compare all tensors
npz_compare.py \
    efficientnet_tensor_all_int8.npz \
    efficientnet_cmdbuf_out_all_int8_multiplier.npz \
    --op_info efficientnet_b0_op_info.csv

# VERDICT
echo $0 PASSED
