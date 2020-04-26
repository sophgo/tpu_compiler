#!/bin/bash
set -e

#
# the passobile cmd like $0 net bf16_quant_layers_file
#

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
NET=$1
BF16_QUANT_LAYERS_FILE=$2

#
# start gen cmd
#

# import calibration table
mlir-opt \
    --import-calibration-table \
    --calibration-table ${NET}_preprocess_calibration_table \
    ${NET}_opt.mlir \
    -o ${NET}_cali.mlir


mlir-opt \
    --tpu-quant-clip \
    ${NET}_cali.mlir \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
    -o ${NET}_quant_int8_multiplier_0.mlir

# quant
echo "quant bf16 layers:"
cat ${BF16_QUANT_LAYERS_FILE}
mlir-opt \
    --tpu-quant \
    --quant-int8-mix-bf16-layers-from-file ${BF16_QUANT_LAYERS_FILE} \
    --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
    --print-tpu-op-info \
    ${NET}_quant_int8_multiplier_0.mlir \
    -o ${NET}_quant_int8_multiplier.mlir
#
##  Lower for quantization
#mlir-opt \
#    --tpu-lower \
#    ${NET}_quant_int8_multiplier.mlir \
#    -o ${NET}_quant_int8_multiplier_tg.mlir
#
## FORCE do MemRefType cuz not back compatible
## function argument lower to MemRefType
#mlir-opt \
#    --convert-func-to-memref \
#    ${NET}_quant_int8_multiplier_tg.mlir \
#    -o ${NET}_quant_int8_multiplier_tg_opt_memref.mlir
#
## op lower to MemRefType
#mlir-opt \
#  --convert-tg-op-to-memref \
#  ${NET}_quant_int8_multiplier_tg_opt_memref.mlir \
#  -o ${NET}_quant_int8_multiplier_tg_opt_op_memref.mlir
#
## memory space w/ global memory reuse
#mlir-opt \
#    --enable-reuse-global-memory=false \
#    --assign-neuron-address-memref \
#    --tpu-neuron-address-align-memref=16 \
#    --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
#    ${NET}_quant_int8_multiplier_tg_opt_op_memref.mlir \
#    -o ${NET}_quant_int8_multiplier_tg_opt_op_memref_addr.mlir
#
## tg op back to TensorType
#mlir-opt \
#    --convert-tg-op-to-tensor \
#    ${NET}_quant_int8_multiplier_tg_opt_op_memref_addr.mlir \
#    -o ${NET}_quant_int8_multiplier_tg_opt_op_tensor_addr.mlir
#
## function argument back to TensorType
#mlir-opt \
#    --convert-func-to-tensor \
#    ${NET}_quant_int8_multiplier_tg_opt_op_tensor_addr.mlir \
#    -o ${NET}_quant_int8_multiplier_tg_opt_addr.mlir
#
## assign weight address & neuron address
#mlir-opt \
#    --assign-weight-address \
#    --tpu-weight-address-align=16 \
#    --tpu-weight-map-filename=${NET}_weight_map_int8_multiplier.csv \
#    --tpu-weight-bin-filename=weight_int8_multiplier.bin \
#    --assign-neuron-address \
#    --tpu-neuron-address-align=16 \
#    --tpu-neuron-map-filename=${NET}_neuron_map_int8_multiplier.csv \
#    --convert-cpu-op \
#    ${NET}_quant_int8_multiplier_tg_opt_addr.mlir \
#    -o ${NET}_quant_int8_multiplier_addr.mlir
#
#mlir-translate \
#  --mlir-to-cmdbuf \
#  ${NET}_quant_int8_multiplier_addr.mlir \
#  -o cmdbuf_int8_multiplier.bin
#
## generate cvimodel
#build_cvimodel.py \
#  --cmdbuf cmdbuf_int8_multiplier.bin \
#  --weight weight_int8_multiplier.bin \
#  --mlir ${NET}_quant_int8_multiplier_addr.mlir \
#  --output=${NET}_int8_multiplier.cvimodel

# VERDICT
echo $0 PASSED
