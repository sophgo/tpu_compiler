#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

CUSTOM_OP_PLUGIN_OPTION=""
if [[ ! -z $CUSTOM_OP_PLUGIN ]]; then
    CUSTOM_OP_PLUGIN_OPTION="--custom-op-plugin ${CUSTOM_OP_PLUGIN}"
fi

if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then
  ###############################################################################
  # quantization 1: per-layer int8
  ###############################################################################
  mlir-opt \
      --tpu-quant --quant-int8-per-tensor \
      ${CUSTOM_OP_PLUGIN_OPTION}\
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_per_tensor.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_int8_per_tensor.mlir

  mlir-tpu-interpreter ${NET}_quant_int8_per_tensor.mlir \
      ${CUSTOM_OP_PLUGIN_OPTION}\
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_per_tensor.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_per_tensor.npz

  if [ $COMPARE_ALL -eq 1 ]; then
    # this will fail for now, because prob has been dequantized twice, others should pass
    cvi_npz_tool.py compare \
        ${NET}_tensor_all_int8_per_tensor.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_per_tensor.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance $TOLERANCE_INT8_PER_TENSOR -vv
  fi


  ################################
  # Lower for quantization 1: per-layer int8
  ################################
  mlir-opt \
      --tpu-lower \
      ${NET}_quant_int8_per_tensor.mlir \
      -o ${NET}_quant_int8_per_tensor_tg.mlir

  mlir-opt \
      ${MLIR_OPT_BE} \
      ${NET}_quant_int8_per_tensor_tg.mlir \
      -o ${NET}_quant_int8_per_tensor_tg_opt.mlir

  # assign weight address & neuron address
  mlir-opt \
      --assign-weight-address \
      --tpu-weight-address-align=16 \
      --tpu-weight-map-filename=${NET}_weight_map_int8_per_tensor.csv \
      --tpu-weight-bin-filename=weight_int8_per_tensor.bin \
      --assign-neuron-address \
      --tpu-neuron-address-align=16 \
      --tpu-neuron-map-filename=${NET}_neuron_map_int8_per_tensor.csv \
      ${NET}_quant_int8_per_tensor_tg_opt.mlir \
      -o ${NET}_quant_int8_per_tensor_addr.mlir

  # cat for logging
  echo "cat ${NET}_quant_int8_per_tensor_addr.mlir"
  cat ${NET}_quant_int8_per_tensor_addr.mlir

  mlir-opt \
      --divide-ops-to-func \
      ${NET}_quant_int8_per_tensor_addr.mlir \
      -o ${NET}_quant_int8_per_tensor_addr_func.mlir

  mlir-translate \
      --mlir-to-cvimodel \
      --cvi-set-chip ${SET_CHIP_NAME} \
      ${CUSTOM_OP_PLUGIN_OPTION}\
      --weight-file weight_int8_per_tensor.bin \
      ${NET}_quant_int8_per_tensor_addr_func.mlir \
      -o ${NET}_int8_per_tensor.cvimodel

  # run cvimodel
  model_runner \
      --dump-all-tensors \
      --input ${NET}_in_fp32.npz \
      --model ${NET}_int8_per_tensor.cvimodel \
      --batch-num $BATCH_SIZE \
      --set-chip ${SET_CHIP_NAME} \
      --output ${NET}_cmdbuf_out_all_int8_per_tensor.npz

  # compare all tensors
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_per_tensor.npz \
      ${NET}_tensor_all_int8_per_tensor.npz \
      --op_info ${NET}_op_info_int8_per_tensor.csv

fi

echo $0 PASSED
