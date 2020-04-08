#!/bin/bash
set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"


# quantization
if [ $DO_QUANT_BF16 -eq 1 ]; then
  mlir-opt \
    --tpu-quant --quant-full-bf16 \
    --print-tpu-op-info \
    --tpu-op-info-filename ${NET}_op_info_bf16.csv \
    ${NET}_opt.mlir \
    -o ${NET}_quant_bf16.mlir

  # bf16 inference
  mlir-tpu-interpreter ${NET}_quant_bf16.mlir \
    --tensor-in ${NET}_in_fp32.npz \
    --tensor-out ${NET}_out_bf16.npz \
    --dump-all-tensor=${NET}_tensor_all_bf16.npz

  cvi_npz_tool.py compare \
    ${NET}_out_bf16.npz \
    ${NET}_out_fp32.npz \
    --tolerance $TOLERANCE_BF16 -vv

  cvi_npz_tool.py compare \
    ${NET}_tensor_all_bf16.npz \
    ${NET}_tensor_all_fp32.npz \
    --op_info ${NET}_op_info_bf16.csv \
    --tolerance $TOLERANCE_BF16 -vv

  if [ $DO_CMDBUF_BF16 -eq 1 ]; then
    ################################
    # Lower
    ################################
    mlir-opt \
      --tpu-lower \
      ${NET}_quant_bf16.mlir \
      -o ${NET}_quant_bf16_tg.mlir

    # assign weight address & neuron address
    mlir-opt \
      --assign-weight-address \
      --tpu-weight-address-align=16 \
      --tpu-weight-map-filename=${NET}_weight_map_bf16.csv \
      --tpu-weight-bin-filename=weight_bf16.bin \
      --assign-neuron-address \
      --tpu-neuron-address-align=16 \
      --tpu-neuron-map-filename=${NET}_neuron_map_bf16.csv \
      --convert-cpu-op \
      ${NET}_quant_bf16_tg.mlir \
      -o ${NET}_quant_bf16_addr.mlir

    # backend translate into cmdbuf
    mlir-translate \
      --mlir-to-cmdbuf \
      ${NET}_quant_bf16_addr.mlir \
      -o cmdbuf_bf16.bin

    # generate cvimodel
    build_cvimodel.py \
      --cmdbuf cmdbuf_bf16.bin \
      --weight weight_bf16.bin \
      --mlir ${NET}_quant_bf16_addr.mlir \
      --output=${NET}_bf16.cvimodel

    # run cvimodel
    model_runner \
      --dump-all-tensors \
      --input ${NET}_in_fp32.npz \
      --model ${NET}_bf16.cvimodel \
      --batch-num $BATCH_SIZE \
      --output ${NET}_cmdbuf_out_all_bf16.npz

    # compare all tensors
    cvi_npz_tool.py compare \
      ${NET}_cmdbuf_out_all_bf16.npz \
      ${NET}_tensor_all_bf16.npz \
      --op_info ${NET}_op_info_bf16.csv \
      --tolerance=$TOLERANCE_BF16_CMDBUF -vv
  fi
fi
# VERDICT
echo $0 PASSED
