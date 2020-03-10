#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

################################
# prepare int8 input
################################
npz_to_bin.py \
    ${NET}_tensor_all_int8_multiplier.npz \
    data \
    ${NET}_in_int8.bin \
    int8

################################
# Lower for quantization 1: per-layer int8
################################
if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then

  mlir-opt \
      --tpu-lower \
      ${NET}_quant_int8_per_tensor.mlir \
      -o ${NET}_quant_int8_per_tensor_tg.mlir

  # assign weight address & neuron address
  mlir-opt \
      --assign-weight-address \
      --tpu-weight-address-align=16 \
      --tpu-weight-map-filename=${NET}_weight_map_int8_per_tensor.csv \
      --tpu-weight-bin-filename=weight_int8_per_tensor.bin \
      --assign-neuron-address \
      --tpu-neuron-address-align=16 \
      --tpu-neuron-map-filename=${NET}_neuron_map_int8_per_tensor.csv \
      ${NET}_quant_int8_per_tensor_tg.mlir \
      -o ${NET}_quant_int8_per_tensor_addr.mlir

  mlir-translate \
      --mlir-to-cmdbuf \
      ${NET}_quant_int8_per_tensor_addr.mlir \
      -o cmdbuf_int8_per_tensor.bin

  # generate cvi model
  build_cvimodel.py \
      --cmdbuf cmdbuf_int8_per_tensor.bin \
      --weight weight_int8_per_tensor.bin \
      --mlir ${NET}_quant_int8_per_tensor_addr.mlir \
      --output=${NET}_int8_per_tensor.cvimodel

  # run cvimodel
  model_runner \
      --dump-all-tensors \
      --input ${NET}_in_int8.bin \
      --model ${NET}_int8_per_tensor.cvimodel \
      --output ${NET}_cmdbuf_out_all_int8_per_tensor.npz

  npz_to_bin.py \
      ${NET}_cmdbuf_out_all_int8_per_tensor.npz \
      ${OUTPUTS} \
      ${NET}_cmdbuf_out_${OUTPUTS}_int8_per_tensor.bin \
      int8
  bin_compare.py \
      ${NET}_cmdbuf_out_${OUTPUTS}_int8_per_tensor.bin \
      ${NET}_out_${OUTPUTS}_int8_per_tensor.bin \
      int8 1 1 1 1000 5

  # compare all tensors
  npz_compare.py \
      ${NET}_cmdbuf_out_all_int8_per_tensor.npz \
      ${NET}_tensor_all_int8_per_tensor.npz \
      --op_info ${NET}_op_info_int8_per_tensor.csv

fi

################################
# Lower for quantization 2: per-channel int8
################################

# skipped

################################
# Lower for quantization 3: multiplier int8
################################
if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then

  mlir-opt \
      --tpu-lower \
      ${NET}_quant_int8_multiplier.mlir \
      -o ${NET}_quant_int8_multiplier_tg.mlir

  # assign weight address & neuron address
  mlir-opt \
      --assign-weight-address \
      --tpu-weight-address-align=16 \
      --tpu-weight-map-filename=${NET}_weight_map_int8_multiplier.csv \
      --tpu-weight-bin-filename=weight_int8_multiplier.bin \
      --assign-neuron-address \
      --tpu-neuron-address-align=16 \
      --tpu-neuron-map-filename=${NET}_neuron_map_int8_multiplier.csv \
      ${NET}_quant_int8_multiplier_tg.mlir \
      -o ${NET}_quant_int8_multiplier_addr.mlir

  mlir-translate \
      --mlir-to-cmdbuf \
      ${NET}_quant_int8_multiplier_addr.mlir \
      -o cmdbuf_int8_multiplier.bin

  # generate cvimodel
  build_cvimodel.py \
      --cmdbuf cmdbuf_int8_multiplier.bin \
      --weight weight_int8_multiplier.bin \
      --mlir ${NET}_quant_int8_multiplier_addr.mlir \
      --output=${NET}_int8_multiplier.cvimodel

  # run cvimodel
  model_runner \
      --dump-all-tensors \
      --input ${NET}_in_int8.bin \
      --model ${NET}_int8_multiplier.cvimodel \
      --output ${NET}_cmdbuf_out_all_int8_multiplier.npz

  npz_to_bin.py \
      ${NET}_cmdbuf_out_all_int8_multiplier.npz \
      ${OUTPUTS} \
      ${NET}_cmdbuf_out_${OUTPUTS}_int8_multiplier.bin \
      int8
  bin_compare.py \
      ${NET}_cmdbuf_out_${OUTPUTS}_int8_multiplier.bin \
      ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
      int8 1 1 1 1000 5

  # compare all tensors
  npz_compare.py \
      ${NET}_cmdbuf_out_all_int8_multiplier.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info ${NET}_op_info_int8_multiplier.csv

  if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
    cp ${NET}_in_int8.bin $CVIMODEL_REL_PATH
    cp ${NET}_int8_multiplier.cvimodel $CVIMODEL_REL_PATH
    cp ${NET}_tensor_all_int8_multiplier.npz $CVIMODEL_REL_PATH
    cp ${NET}_neuron_map_int8_multiplier.csv $CVIMODEL_REL_PATH
  fi

fi

# VERDICT
echo $0 PASSED
