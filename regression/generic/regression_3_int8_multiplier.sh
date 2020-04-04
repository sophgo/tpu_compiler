#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1
COMPARE_OUTPUT_BIT_TRUE=0

OP_LOWERING=0

if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then
  ###############################################################################
  # quantization 3: per-channel int8 with multiplier
  ###############################################################################
  mlir-opt \
      --tpu-quant \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_int8_multiplier.mlir

  mlir-tpu-interpreter ${NET}_quant_int8_multiplier.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_multiplier.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_multiplier.npz

  if [ $COMPARE_OUTPUT_BIT_TRUE -eq 1 ]; then
    cvi_npz_tool.py to_bin \
        ${NET}_tensor_all_int8_multiplier.npz \
        ${OUTPUTS} \
        ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
        int8
    bin_compare.py \
        ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
        $REGRESSION_PATH/${NET}/data/test_cat_out_${NET}_${OUTPUTS}_int8_multiplier.bin \
        int8 ${BATCH_SIZE} 1 1 1000 5
  fi

  if [ $COMPARE_ALL -eq 1 ]; then
    # this will fail for now, because prob has been dequantized twice, others should pass
    cvi_npz_tool.py compare \
        ${NET}_tensor_all_int8_multiplier.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_multiplier.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance $TOLERANCE_INT8_MULTIPLER -vv \
        --stats_int8_tensor
  fi

  ################################
  # Lower for quantization 3: multiplier int8
  ################################
  mlir-opt \
      --tpu-lower \
      ${NET}_quant_int8_multiplier.mlir \
      -o ${NET}_quant_int8_multiplier_tg.mlir

  mlir-opt \
      ${MLIR_OPT_BE} \
      ${NET}_quant_int8_multiplier_tg.mlir \
      -o ${NET}_quant_int8_multiplier_tg_opt.mlir

  if [ $OP_LOWERING -eq 1 ]; then
    # function argument lower to MemRefType
    mlir-opt \
        --convert-func-to-memref \
        ${NET}_quant_int8_multiplier_tg_opt.mlir \
        -o ${NET}_quant_int8_multiplier_tg_opt_memref.mlir

    # op lower to MemRefType
    mlir-opt \
      --convert-tg-op-to-memref \
      ${NET}_quant_int8_multiplier_tg_opt_memref.mlir \
      -o ${NET}_quant_int8_multiplier_tg_opt_op_memref.mlir

    # memory space w/ global memory reuse
    mlir-opt \
        --enable-reuse-global-memory=true \
        --assign-neuron-address-memref \
        --tpu-neuron-address-align-memref=16 \
        --tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv \
        ${NET}_quant_int8_multiplier_tg_opt_op_memref.mlir \
        -o ${NET}_quant_int8_multiplier_tg_opt_op_memref_addr.mlir

    # tg op back to TensorType
    mlir-opt \
        --convert-tg-op-to-tensor \
        ${NET}_quant_int8_multiplier_tg_opt_op_memref_addr.mlir \
        -o ${NET}_quant_int8_multiplier_tg_opt_op_tensor_addr.mlir

    # function argument back to TensorType
    mlir-opt \
        --convert-func-to-tensor \
        ${NET}_quant_int8_multiplier_tg_opt_op_tensor_addr.mlir \
        -o ${NET}_quant_int8_multiplier_tg_opt_addr.mlir

    # assign weight address & neuron address
    mlir-opt \
        --assign-weight-address \
        --tpu-weight-address-align=16 \
        --tpu-weight-map-filename=${NET}_weight_map_int8_multiplier.csv \
        --tpu-weight-bin-filename=weight_int8_multiplier.bin \
        --assign-neuron-address \
        --tpu-neuron-address-align=16 \
        --tpu-neuron-map-filename=${NET}_neuron_map_int8_multiplier.csv \
        ${NET}_quant_int8_multiplier_tg_opt_addr.mlir \
        -o ${NET}_quant_int8_multiplier_addr.mlir
  else
    # assign weight address & neuron address
    mlir-opt \
        --assign-weight-address \
        --tpu-weight-address-align=16 \
        --tpu-weight-map-filename=${NET}_weight_map_int8_multiplier.csv \
        --tpu-weight-bin-filename=weight_int8_multiplier.bin \
        --assign-neuron-address \
        --tpu-neuron-address-align=16 \
        --tpu-neuron-map-filename=${NET}_neuron_map_int8_multiplier.csv \
        ${NET}_quant_int8_multiplier_tg_opt.mlir \
        -o ${NET}_quant_int8_multiplier_addr.mlir
  fi

  # cat for logging
  echo "cat ${NET}_quant_int8_multiplier_addr.mlir"
  cat ${NET}_quant_int8_multiplier_addr.mlir

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
      --input ${NET}_in_fp32.npz \
      --model ${NET}_int8_multiplier.cvimodel \
      --batch-num $BATCH_SIZE \
      --output ${NET}_cmdbuf_out_all_int8_multiplier.npz

  #cvi_npz_tool.py to_bin \
  #    ${NET}_cmdbuf_out_all_int8_multiplier.npz \
  #    ${OUTPUTS} \
  #    ${NET}_cmdbuf_out_${OUTPUTS}_int8_multiplier.bin \
  #    int8
  #bin_compare.py \
  #    ${NET}_cmdbuf_out_${OUTPUTS}_int8_multiplier.bin \
  #    ${NET}_out_${OUTPUTS}_int8_multiplier.bin \
  #    int8 ${BATCH_SIZE} 1 1 1000 5

  # compare all tensors
  cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_int8_multiplier.npz \
      ${NET}_tensor_all_int8_multiplier.npz \
      --op_info ${NET}_op_info_int8_multiplier.csv

  if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
    cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
    mv ${NET}_int8_multiplier.cvimodel $CVIMODEL_REL_PATH
    cp ${NET}_cmdbuf_out_all_int8_multiplier.npz $CVIMODEL_REL_PATH
  fi

fi

# VERDICT
echo $0 PASSED
