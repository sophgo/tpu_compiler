#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


COMPARE_ALL=1

CUSTOM_OP_PLUGIN_OPTION=""
if [[ ! -z $CUSTOM_OP_PLUGIN ]]; then
    CUSTOM_OP_PLUGIN_OPTION="--custom-op-plugin ${CUSTOM_OP_PLUGIN}"
fi

if [ $DO_QUANT_INT8_RFHIFT_ONLY -eq 1 ]; then
  ###############################################################################
  # quantization 2: per-channel(rshift_only) int8
  ###############################################################################
  mlir-opt \
      --assign-chip-name \
      --chipname ${SET_CHIP_NAME} \
      ${CUSTOM_OP_PLUGIN_OPTION} \
      --tpu-quant --quant-int8-rshift-only \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_int8_rshift_only.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_int8_rshift_only.mlir

  mlir-tpu-interpreter ${NET}_quant_int8_rshift_only.mlir \
      ${CUSTOM_OP_PLUGIN_OPTION}\
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_int8_rshift_only.npz \
      --dump-all-tensor=${NET}_tensor_all_int8_rshift_only.npz

  if [ $COMPARE_ALL -eq 1 ]; then
    # this will fail for now, because prob has been dequantized twice, others should pass
    cvi_npz_tool.py compare \
        ${NET}_tensor_all_int8_rshift_only.npz \
        ${NET}_blobs.npz \
        --op_info ${NET}_op_info_int8_rshift_only.csv \
        --dequant \
        --excepts $EXCEPTS \
        --tolerance $TOLERANCE_INT8_RSHIFT_ONLY -vv
  fi
  ################################
  # Lower for quantization
  ################################
  # skipped
fi

echo $0 PASSED
