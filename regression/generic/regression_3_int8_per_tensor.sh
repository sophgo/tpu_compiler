#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1
COMPARE_OUTPUT_BIT_TRUE=0

if [ $DO_QUANT_INT8_PER_TENSOR -eq 1 ]; then
  ###############################################################################
  # quantization 1: per-layer int8
  ###############################################################################
  mlir-opt \
      --tpu-quant --quant-int8-per-tensor \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_per_tensor.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_int8_per_tensor.mlir

  mlir-tpu-interpreter ${NET}_quant_int8_per_tensor.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_per_tensor.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_per_tensor.npz

  if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
    cvi_npz_tool.py to_bin \
        ${NET}_tensor_all_int8_per_tensor.npz \
        ${OUTPUTS} \
        ${NET}_out_${OUTPUTS}_int8_per_tensor.bin \
        int8

    bin_compare.py \
        ${NET}_out_${OUTPUTS}_int8_per_tensor.bin \
        $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_per_tensor.bin \
        int8 ${BATCH_SIZE} 1 1 1000 5
  fi

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
      --convert-cpu-op \
      ${NET}_quant_int8_per_tensor_tg_opt.mlir \
      -o ${NET}_quant_int8_per_tensor_addr.mlir

  # cat for logging
  echo "cat ${NET}_quant_int8_per_tensor_addr.mlir"
  cat ${NET}_quant_int8_per_tensor_addr.mlir

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
      --input ${NET}_in_fp32.npz \
      --model ${NET}_int8_per_tensor.cvimodel \
      --batch-num $BATCH_SIZE \
      --output ${NET}_cmdbuf_out_all_int8_per_tensor.npz

  # compare all tensors
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_per_tensor.npz \
      ${NET}_tensor_all_int8_per_tensor.npz \
      --op_info ${NET}_op_info_int8_per_tensor.csv

fi

echo $0 PASSED
