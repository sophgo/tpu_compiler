#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
source $DIR/../../envsetup.sh

COMPARE_ALL=1

###############################################################################
# Mix-Precison 1: mix-bf16-broadcastmul + mix-bf16-sigmoid mix-bf16-eltwisemul
###############################################################################
if [ $DO_QUANT_MIX -eq 1 ]; then

  mlir-opt \
      --tpu-quant \
      --quant-int8-mix-bf16-sigmoid \
      --quant-int8-mix-bf16-broadcastmul \
      --quant-int8-mix-bf16-eltwisemul \
      --print-tpu-op-info \
      --tpu-op-info-filename ${NET}_op_info_mix.csv \
      ${NET}_cali.mlir \
      -o ${NET}_quant_mix.mlir

  mlir-tpu-interpreter ${NET}_quant_mix.mlir \
      --tensor-in ${NET}_in_fp32.npz \
      --tensor-out ${NET}_out_mix.npz \
      --dump-all-tensor=${NET}_tensor_all_mix.npz

  npz_compare.py \
      ${NET}_tensor_all_mix.npz \
      ${NET}_blobs.npz \
      --op_info ${NET}_op_info_mix.csv \
      --dequant \
      --excepts $EXCEPTS \
      --tolerance $TOLERANCE_MIX -vv \
      --stats_int8_tensor

fi

# VERDICT
echo $0 PASSED
