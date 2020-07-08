#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

COMPRESS_ACTIVATION=0

CUSTOM_OP_PLUGIN_OPTION=""
if [[ ! -z $CUSTOM_OP_PLUGIN ]]; then
    CUSTOM_OP_PLUGIN_OPTION="--custom-op-plugin ${CUSTOM_OP_PLUGIN}"
fi

if [ $DO_QUANT_INT8_MULTIPLER -eq 1 ]; then
  ###############################################################################
  # quantization 3: per-channel int8 with multiplier
  ###############################################################################
  if [ $DO_PREPROCESS -eq 1 ]; then
    mlir-opt \
        --assign-chip-name \
        --chipname ${SET_CHIP_NAME} \
        ${CUSTOM_OP_PLUGIN_OPTION} \
        --tpu-quant \
        --convert-quant-op \
        --print-tpu-op-info \
        --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
        ${NET}_cali.mlir \
        -o ${NET}_quant_int8_multiplier.mlir
  else
    mlir-opt \
        --assign-chip-name \
        --chipname ${SET_CHIP_NAME} \
        ${CUSTOM_OP_PLUGIN_OPTION} \
        --tpu-quant \
        --print-tpu-op-info \
        --tpu-op-info-filename ${NET}_op_info_int8_multiplier.csv \
        ${NET}_cali.mlir \
        -o ${NET}_quant_int8_multiplier.mlir
  fi

  mlir-tpu-interpreter ${NET}_quant_int8_multiplier.mlir \
      ${CUSTOM_OP_PLUGIN_OPTION} \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_multiplier.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_multiplier.npz

  if [ $COMPARE_ALL -eq 1 ]; then
    # this will fail for now, because prob has been dequantized twice, others should pass
    cvi_npz_tool.py compare \
        ${NET}_tensor_all_int8_multiplier.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_multiplier.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance="$TOLERANCE_INT8_MULTIPLER" \
        -vv \
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
      --reorder-op \
      ${NET}_quant_int8_multiplier_tg.mlir \
      -o ${NET}_quant_int8_multiplier_tg_reorder.mlir

  mlir-opt \
      ${MLIR_OPT_BE} \
      ${NET}_quant_int8_multiplier_tg_reorder.mlir \
      -o ${NET}_quant_int8_multiplier_tg_opt.mlir

  if [ $COMPRESS_ACTIVATION -eq 1 ]; then
    mlir-opt \
        --debug \
        --compress-activation \
        ${NET}_quant_int8_multiplier_tg_opt.mlir \
        -o ${NET}_quant_int8_multiplier_tg_opt_ca.mlir
    mv ${NET}_quant_int8_multiplier_tg_opt_ca.mlir ${NET}_quant_int8_multiplier_tg_opt.mlir
  fi

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


  # cat for logging
  # echo "cat ${NET}_quant_int8_multiplier_addr.mlir"
  # cat ${NET}_quant_int8_multiplier_addr.mlir

  mlir-opt \
      --divide-ops-to-func \
      ${NET}_quant_int8_multiplier_addr.mlir \
      -o ${NET}_quant_int8_multiplier_addr_func.mlir

  mlir-translate \
      --mlir-to-cvimodel \
      ${CUSTOM_OP_PLUGIN_OPTION}\
      --weight-file weight_int8_multiplier.bin \
      ${NET}_quant_int8_multiplier_addr_func.mlir \
      -o ${NET}_int8_multiplier.cvimodel

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

  # if [ ! -z $CVIMODEL_REL_PATH -a -d $CVIMODEL_REL_PATH ]; then
  #   cp ${NET}_in_fp32.npz $CVIMODEL_REL_PATH
  #   mv ${NET}_int8_multiplier.cvimodel $CVIMODEL_REL_PATH
  #   cp ${NET}_cmdbuf_out_all_int8_multiplier.npz $CVIMODEL_REL_PATH
  # fi

fi

# VERDICT
echo $0 PASSED
