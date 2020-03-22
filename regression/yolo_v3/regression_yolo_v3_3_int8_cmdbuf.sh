#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=0

################################
# prepare int8 input
################################

cvi_npz_tool.py to_bin \
    ${NET}_tensor_all_int8_multiplier.npz \
    data \
    ${NET}_in_int8.bin \
    int8

# don't use following commands to generate input, as it depends on
# calibration result.
# cvi_npz_tool.py to_bin ${NET}_in_fp32.npz input ${NET}_in_fp32.bin
# bin_fp32_to_int8.py \
#     ${NET}_in_fp32.bin \
#     ${NET}_in_int8.bin \
#     1.0 \
#     1.00000488758

################################
# Lower for quantization 1: per-layer int8
################################
mlir-opt \
    --tpu-lower \
    ${NET}_quant_int8_per_layer.mlir \
    -o ${NET}_quant_int8_per_layer_tg.mlir

# apply all possible backend optimizations
mlir-opt \
    --tg-fuse-leakyrelu \
    ${NET}_quant_int8_per_layer_tg.mlir \
    -o ${NET}_quant_int8_per_layer_tg_opt.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_per_layer.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    ${NET}_quant_int8_per_layer_tg_opt.mlir \
    -o ${NET}_quant_int8_per_layer_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_int8_per_layer_addr.mlir \
    -o cmdbuf_int8_per_layer.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_per_layer.bin \
    --weight weight_int8_per_layer.bin \
    --mlir ${NET}_quant_int8_per_layer_addr.mlir \
    --output=${NET}_int8_per_layer.cvimodel

# run cmdbuf
model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_per_layer.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8_per_layer.npz

cvi_npz_tool.py extract \
    ${NET}_cmdbuf_out_all_int8_per_layer.npz \
    ${NET}_out_int8_three_layer.npz \
    layer82-conv,layer94-conv,layer106-conv

cvi_npz_tool.py compare \
      ${NET}_out_int8_three_layer.npz \
      ${NET}_tensor_all_int8_per_layer.npz \
      --op_info ${NET}_op_info_int8_per_layer.csv

if [ $COMPARE_ALL -eq 1 ]; then
  # some are not equal due to fusion
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_per_layer.npz \
      ${NET}_tensor_all_int8_per_layer.npz \
      --op_info ${NET}_op_info_int8_per_layer.csv
fi

################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
mlir-opt \
    --tpu-lower \
    ${NET}_quant_int8_multiplier.mlir \
    -o ${NET}_quant_int8_multiplier_tg.mlir

# apply all possible backend optimizations
mlir-opt \
    --tg-fuse-leakyrelu \
    ${NET}_quant_int8_multiplier_tg.mlir \
    -o ${NET}_quant_int8_multiplier_tg_opt.mlir

# assign weight address & neuron address
mlir-opt \
    --assign-weight-address \
    --tpu-weight-address-align=16 \
    --tpu-weight-map-filename=weight_map.csv \
    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
    --assign-neuron-address \
    --tpu-neuron-address-align=16 \
    --tpu-neuron-map-filename=neuron_map.csv \
    ${NET}_quant_int8_multiplier_tg_opt.mlir \
    -o ${NET}_quant_int8_multiplier_addr.mlir

mlir-translate \
    --mlir-to-cmdbuf \
    ${NET}_quant_int8_multiplier_addr.mlir \
    -o cmdbuf_int8_multiplier.bin

# generate cvi model
build_cvimodel.py \
    --cmdbuf cmdbuf_int8_multiplier.bin \
    --weight weight_int8_multiplier.bin \
    --mlir ${NET}_quant_int8_multiplier_addr.mlir \
    --output=${NET}_int8_multiplier.cvimodel

# run cmdbuf
#$RUNTIME_PATH/bin/test_bmnet \
#    ${NET}_in_int8.bin \
#    weight_int8_multiplier.bin \
#    cmdbuf_int8_multiplier.bin \
#    ${NET}_cmdbuf_out_all_int8_multiplier.bin \
#    94614832 0 94614832 1
model_runner \
    --dump-all-tensors \
    --input ${NET}_in_fp32.npz \
    --model ${NET}_int8_multiplier.cvimodel \
    --output ${NET}_cmdbuf_out_all_int8_multiplier.npz

cvi_npz_tool.py extract \
    ${NET}_cmdbuf_out_all_int8_multiplier.npz \
    ${NET}_out_int8_multiplier_three_layer.npz \
    layer82-conv,layer94-conv,layer106-conv

cvi_npz_tool.py compare \
      ${NET}_out_int8_multiplier_three_layer.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info ${NET}_op_info_int8_per_layer.csv

if [ $COMPARE_ALL -eq 1 ]; then
  # some are not equal due to fusion
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_multiplier.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info ${NET}_op_info_int8_per_layer.csv
fi

if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
  cp ${NET}_int8_multiplier.cvimodel $CVIMODEL_REL_PATH
  cp ${NET}_cmdbuf_out_all_int8_multiplier.npz $CVIMODEL_REL_PATH
  # cp ${NET}_tensor_all_int8_multiplier.npz $CVIMODEL_REL_PATH
  # cp ${NET}_neuron_map_int8_multiplier.csv $CVIMODEL_REL_PATH
fi

# VERDICT
echo $0 PASSED
